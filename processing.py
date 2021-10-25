# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 16:26:32 2021

@author: balakin
"""

from collections import defaultdict
from os import remove
from time import perf_counter
import logging
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from scipy.fft import dctn, idctn #pylint: disable=E0611
from scipy.stats.qmc import Sobol
from haar_transform import haar_transform, haar_transform_2d, inverse_haar_transform, inverse_haar_transform_2d
from scipy.sparse.linalg import lsmr
from imageio import imread
import cvxpy as cp
from cvxpy.atoms.affine.sum import sum as cp_sum
from cvxpy.atoms.affine.diff import diff as cp_diff
from cvxpy.atoms.affine.reshape import reshape as cp_reshape
from cvxpy.atoms import norm as cp_norm
from cvxpy.atoms.norm1 import norm1 as cp_norm1
import matplotlib.pyplot as plt
from tqdm import tqdm
from misc import load_demo_image, save_image_for_show
from fiber_propagation import propagator
from reduction import dense_reduction, sparse_reduction, dense_reduction_iter


logger = logging.getLogger("Fiber-GI-reduction")
logger.propagate = False # Do not propagate to the root logger
logger.setLevel(logging.INFO)
logger.handlers = []
fh = logging.FileHandler("processing.log")
fh.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)


def synth(measurement, mt_op, img_shape, noise_var, omega):
    tmp_op = mt_op.dot(mt_op.T).astype(float)
    tmp_op[np.diag_indices_from(tmp_op)] += noise_var*omega
    return mt_op.T.dot(np.linalg.solve(tmp_op, measurement)).reshape(img_shape)


def compressive_l1(measurement, mt_op, img_shape, alpha=None, full: bool = False):
    tmp_op = mt_op.T.reshape(img_shape + (-1,))
    mt_op_dct = dctn(tmp_op, axes=(0, 1), norm="ortho").reshape((img_shape[0]*img_shape[1], -1)).T #pylint: disable=E1101

    dct_f = cp.Variable(mt_op_dct.shape[1])
    sparsity_term = cp_norm1(dct_f)**2
    if alpha is None:
        objective = cp.Minimize(sparsity_term)
        constraints = [mt_op_dct @ dct_f == measurement]
        prob = cp.Problem(objective, constraints)
    else:
        fidelity = cp_sum((measurement - mt_op_dct @ dct_f)**2)
        objective = cp.Minimize(alpha*sparsity_term + fidelity)
        prob = cp.Problem(objective)
    try:
        prob.solve()
    except cp.error.SolverError:
        prob.solve(solver=cp.ECOS)
    result_dct = dct_f.value.reshape(img_shape)
    result = idctn(result_dct, axes=(0, 1), norm="ortho")
    if full and alpha is not None:
        return result, fidelity.value, sparsity_term.value
    else:
        return result


def compressive_l1_haar(measurement, mt_op, img_shape, alpha=None, full=False):
    tmp_op = mt_op.reshape((-1,) + img_shape)
    # Assuming that both elements of img_shape are powers of 2.
    back_transform_matrix = haar_transform(np.eye(img_shape[0]), axis=0,
                                           inplace=True)
    mt_op_haar = np.tensordot(tmp_op, back_transform_matrix, axes=(1, 0))
    if img_shape[1] != img_shape[0]:
        back_transform_matrix = haar_transform(np.eye(img_shape[1]), axis=0,
                                               inplace=True)
    mt_op_haar = np.tensordot(mt_op_haar, back_transform_matrix, axes=(1, 0))
    mt_op_haar = mt_op_haar.reshape((-1, img_shape[0]*img_shape[1]))

    haar_f = cp.Variable(mt_op_haar.shape[1])
    sparsity_term = cp_norm1(haar_f)**2
    if alpha is None:
        objective = cp.Minimize(sparsity_term)
        constraints = [mt_op_haar @ haar_f == measurement]
        prob = cp.Problem(objective, constraints)
    else:
        fidelity = cp_sum((measurement - mt_op_haar @ haar_f)**2)
        objective = cp.Minimize(alpha*sparsity_term + fidelity)
        prob = cp.Problem(objective)
    try:
        prob.solve()
    except cp.error.SolverError:
        prob.solve(solver=cp.ECOS)
    result_haar = haar_f.value.reshape(img_shape)
    result = inverse_haar_transform_2d(result_haar)
    if full:
        return result, fidelity.value, sparsity_term.value
    else:
        return result


def compressive_tc2(measurement, mt_op, img_shape, alpha=None, full=False):
    f = cp.Variable(mt_op.shape[1])
    f2 = cp_reshape(f, img_shape)
    sparsity_term = (cp_norm(cp_diff(f2, k=2, axis=0), 2)
                     + cp_norm(cp_diff(f2, k=2, axis=1), 2))
    if alpha is None:
        objective = cp.Minimize(sparsity_term)
        constraints = [mt_op @ f == measurement]
        prob = cp.Problem(objective, constraints)
    else:
        fidelity = cp_norm(measurement - mt_op @ f, 2)
        objective = cp.Minimize(alpha*sparsity_term + fidelity)
        prob = cp.Problem(objective)
    prob.solve()
    result = f.value.reshape(img_shape)
    if full and alpha is not None:
        return result, fidelity.value, sparsity_term.value
    else:
        return result


def compressive_tv(measurement, mt_op, img_shape, alpha=None):
    initial_estimate = lsmr(mt_op, measurement)[0]
    beta = 1.

    def sparsity_func(f):
        diff0 = np.diff(f.reshape(img_shape), axis=0, n=1)[:, :-1]
        diff1 = np.diff(f.reshape(img_shape), axis=1, n=1)[:-1, :]
        smoothed_diffs = np.sqrt(beta**2 + diff0**2 + diff1**2)

        tmp0 = diff0/smoothed_diffs
        tmp1 = diff1/smoothed_diffs
        grad = np.zeros_like(f.reshape(img_shape))
        grad[: -1, : -1] = -tmp0 - tmp1
        grad[1:, : -1] += tmp0
        grad[: -1, 1:] += tmp1
        return smoothed_diffs.sum(), grad.ravel()

    if alpha is None:
        con = LinearConstraint(mt_op, measurement, measurement)
        res = minimize(sparsity_func, initial_estimate, constraints=con,
                       jac=True, options={"disp": True})
    else:
        def combined_obj_func(f):
            residual = measurement - mt_op.dot(f)
            main_val = (residual**2).sum()
            main_grad = -2*mt_op.T.dot(residual)
            sparsity_val, sparsity_grad = sparsity_func(f)
            return main_val + alpha*sparsity_val, main_grad + alpha*sparsity_grad

        res = minimize(combined_obj_func, initial_estimate, jac=True, options={"disp": True})

    return res.x.reshape(img_shape)


def compressive_tv_alt(measurement, mt_op, img_shape, alpha=None, full=False):
    f = cp.Variable(mt_op.shape[1])
    f2 = cp_reshape(f, img_shape)
    sparsity_term = (cp_norm1(cp_diff(f2, k=1, axis=0))
                     + cp_norm1(cp_diff(f2, k=1, axis=1)))**2
    if alpha is None:
        objective = cp.Minimize(sparsity_term)
        constraints = [mt_op @ f == measurement]
        prob = cp.Problem(objective, constraints)
    else:
        fidelity = cp_sum((measurement - mt_op @ f)**2)
        objective = cp.Minimize(alpha*sparsity_term + fidelity)
        prob = cp.Problem(objective)
    # prob.solve(solver=cp.ECOS)
    prob.solve()
    result = f.value.reshape(img_shape)
    if full and alpha is not None:
        return result, fidelity.value, sparsity_term.value
    else:
        return result


def compressive_tv_alt2(measurement, mt_op, img_shape, alpha=None, full=False):
    f = cp.Variable(mt_op.shape[1])
    f2 = cp_reshape(f, img_shape)
    sparsity_term = cp.atoms.total_variation.tv(f2)**2
    if alpha is None:
        objective = cp.Minimize(sparsity_term)
        constraints = [mt_op @ f == measurement]
        prob = cp.Problem(objective, constraints)
    else:
        fidelity = cp_sum((measurement - mt_op @ f)**2)
        objective = cp.Minimize(alpha*sparsity_term + fidelity)
        prob = cp.Problem(objective)
    prob.solve(solver=cp.SCS)
    result = f.value.reshape(img_shape)
    if full and alpha is not None:
        return result, fidelity.value, sparsity_term.value
    else:
        return result


def figure_name_format(img_id, noise_var=0., kind="", alpha=None,
                       other_params=None, pattern_type="pseudorandom"):
    name = "{}_{}_{:.0e}_{}".format(img_id, pattern_type[0], noise_var, kind)
    if alpha is not None:
        if alpha != "":
            try:
                name = name + "_{:.0e}".format(alpha)
            except TypeError:
                name = name + "_{}".format(alpha)
    if other_params is not None:
        name = name + "_{}".format(other_params)
    return name


def build_pseudorandom_patterns(n_patterns: int, shape):
    rng = np.random.default_rng(2021)
    propagate_func = propagator(shape[0])
    illum_patterns = rng.integers(0, 1, size=(n_patterns,) + shape,
                                      endpoint=True)
    for i in range(n_patterns):
        illum_patterns[i, ...] = propagate_func(illum_patterns[i, ...])
    return illum_patterns


def build_quasirandom_patterns(n_patterns: int, shape):
    propagate_func = propagator(shape[0])

    gen = Sobol(shape[0]*shape[1], scramble=False, seed=2021).fast_forward(2)
    illum_patterns = (gen.random(n_patterns) >= 0.5).reshape((n_patterns,) + shape).astype(float)

    for i in range(n_patterns):
        illum_patterns[i, ...] = propagate_func(illum_patterns[i, ...])
    return illum_patterns


def load_speckle_patterns(n_patterns: int):
    illum_patterns = []
    logger.debug("Loading speckle patterns")
    try:
        for pattern_no in range(n_patterns):
            illum_patterns.append(imread("speckle_patterns/slm{}.bmp".format(pattern_no), as_gray=True))
    except FileNotFoundError:
        raise ValueError("Not enough speckle pattern data for {} patterns.".format(n_patterns))
    illum_patterns = np.array(illum_patterns)
    return illum_patterns


def prepare_measurements(img_id: int = 3, noise_var: float = 0, n_patterns: int = 1024, pattern_type: str="pseudorandom"):
    if pattern_type == "speckle":
        illum_patterns = load_speckle_patterns(n_patterns)
        src_img = load_demo_image(img_id)
        if illum_patterns.shape[1] != src_img.shape[0]:
            diff = (illum_patterns.shape[1] - src_img.shape[0])//2
            src_img = load_demo_image(img_id, pad_by=diff)
        size = 5.2 * illum_patterns.shape[1]/2 # μm
    else:
        if img_id == 6 or img_id == 7:
            src_img = load_demo_image(img_id)
        else:
            src_img = load_demo_image(img_id, pad_by=32)
        size = 2.5 * 50. / 4

        if pattern_type == "pseudorandom":
            illum_patterns = build_pseudorandom_patterns(n_patterns, src_img.shape)
        elif pattern_type == "quasirandom":
            illum_patterns = build_quasirandom_patterns(n_patterns, src_img.shape)


    mt_op = illum_patterns.reshape((n_patterns, -1))
    measurement = mt_op.dot(src_img.ravel())

    if noise_var > 0:
        rng = np.random.default_rng(2021)
        measurement += rng.normal(scale=noise_var**0.5, size=measurement.shape)

    return mt_op, illum_patterns, measurement, src_img, size


def finding_alpha(img_id: int = 3, noise_var: float = 0, proc_kind: str = "l1") -> None:
    """
    Generate images of processing results by the specified method for some
    values or regularization factor alpha to determine the best value.

    Parameters
    ----------
    img_id : int, optional
        Id of the image of the object to be used for processing. The default is 3.
    noise_var : float, optional
        Noise variance. The default is 0.
    proc_kind : str, optional
        Specification of the processing method. The default is "l1".
        The valid values are "l1", "l1h", "tc2", "tva" and "tva2".

    Returns
    -------
    None
    """
    processing_method = {"l1": compressive_l1, "tc2": compressive_tc2,
                         "tva": compressive_tv_alt, "l1h": compressive_l1_haar,
                         "tva2": compressive_tv_alt2}

    mt_op, _, measurement, src_img, _ = prepare_measurements(
        img_id=img_id, noise_var=noise_var
    )
    # estimate = processing_method[proc_kind](measurement, mt_op, src_img.shape,
    #                                         alpha=None)
    # save_image_for_show(
    #     estimate, figure_name_format(img_id, noise_var, proc_kind, alpha=None),
    #     rescale=True
    # )
    for alpha in [1e-9, 1e-6, 1e-3, 1e-1, 1, 1e1]:
        estimate = processing_method[proc_kind](measurement, mt_op,
                                                src_img.shape, alpha=alpha)
        save_image_for_show(
            estimate, figure_name_format(img_id, noise_var, proc_kind,
                                         alpha=alpha),
            rescale=True
        )
    logger.INFO("Done for imd_id = {}, noise_var = {} and proc_kind = {}".format(
        img_id, noise_var, proc_kind
    ))


def finding_alpha_l_curve(img_id: int = 3, noise_var: float = 0,
                          proc_kind: str = "l1", n_measurements: int = 1024) -> None:
    """
    Save data to plot the L-curve for finding the value of alpha
    to use in image processing.

    Parameters
    ----------
    img_id : int, optional
        Id of the image of the object to be used for processing. The default is 3.
    noise_var : float, optional
        Noise variance. The default is 0.
    proc_kind : str, optional
        Specification of the processing method. The default is "l1".
        The valid values are "l1", "l1h", "tc2", "tva" and "tva2".
    n_measurements : int
        The number of illumination patterns to use.

    Returns
    -------
    None

    """
    processing_method = {"l1": compressive_l1, "tc2": compressive_tc2,
                         "tva": compressive_tv_alt, "l1h": compressive_l1_haar,
                         "tva2": compressive_tv_alt2}[proc_kind]

    mt_op, _, measurement, src_img, _ = prepare_measurements(
        img_id=img_id, noise_var=noise_var
    )

    residuals = []
    reg_terms = []
    alpha_values = np.geomspace(1e-5, 1e2, num=11)
    #TODO Try a parallel execution of the loop body
    # Though it will likely fail due to lack of memory
    for alpha in tqdm(alpha_values):
        _, resid, reg = processing_method(measurement, mt_op, src_img.shape,
                                          alpha=alpha, full=True)
        residuals.append(resid)
        reg_terms.append(reg)

    residuals = np.array(residuals)
    reg_terms = np.array(reg_terms)
    np.savez_compressed(
        "../l_curve/{}_{:.0e}_{}_{}.npz".format(
            img_id, noise_var, proc_kind, n_measurements
        ),
        alpha=alpha_values, resid=residuals, reg=reg_terms
    )


def plot_l_curve(img_id: int = 3, noise_var: float = 0, proc_kind: str = "l1",
                 n_measurements: int = 1024) -> None:
    """
    Plot the L-curve for finding the value of alpha to use in image processing.

    Parameters
    ----------
    img_id : int, optional
        Id of the image of the object to be used for processing. The default is 3.
    noise_var : float, optional
        Noise variance. The default is 0.
    proc_kind : str, optional
        Specification of the processing method. The default is "l1".
        The valid values are "l1", "l1h", "tc2", "tva" and "tva2".
    n_measurements : int
        The number of illumination patterns to use.

    Returns
    -------
    None

    """
    with np.load("../l_curve/{}_{:.0e}_{}_{}.npz".format(
            img_id, noise_var, proc_kind, n_measurements
    )) as data:
        alpha_values = data["alpha"]
        reg_terms = data["reg"]
        residuals = data["resid"]
    plt.loglog(reg_terms, residuals)
    plt.xlabel("regularity")
    plt.ylabel("residual")
    for x, y, alpha in zip(reg_terms, residuals, alpha_values):
        plt.annotate("a = {:.3g}".format(alpha), (x, y), xycoords="data")
    plt.show()


def finding_iter_params(img_id: int = 3, noise_var: float = 0) -> None:
    mt_op, _, measurement, src_img, _ = prepare_measurements(
        img_id=img_id, noise_var=noise_var
    )
    for relax in [1.]:
        if relax == 1.:
            pp = []
        else:
            pp = False
        estimate = dense_reduction_iter(measurement, mt_op, src_img.shape,
                                        relax=relax, n_iter=1000000, print_progress=pp)
        save_image_for_show(estimate, figure_name_format(
            img_id, noise_var, "red-iter", alpha=relax
        ), rescale=True)
        if pp:
            plt.plot(pp)
    logger.info("Done for imd_id = {}, noise_var = {} and proc_kind = red-iter".format(
        img_id, noise_var
    ))
    plt.show()


def show_methods(img_id=3, noise_var=0., n_patterns=1024, save: bool=True, show: bool=True, pattern_type: str="pseudorandom"):
    mt_op, illum_patterns, measurement, src_img, size = prepare_measurements(
        img_id=img_id, noise_var=noise_var,
        n_patterns=n_patterns,
        pattern_type=pattern_type
    )

    traditional_gi = np.tensordot(measurement - measurement.mean(),
                                  illum_patterns - illum_patterns.mean(axis=0),
                                  axes=1)/measurement.size

    # whitened_measurement = noise_var**0.5 * measurement

    # left_sing, sing_val, right_sing = np.linalg.svd(noise_var**0.5 * mt_op,
    #                                                 full_matrices=False)

    # sing_val2 = 1/sing_val
    # rxi = (right_sing.T * sing_val2) @ left_sing.T @ whitened_measurement

    # eigv_num = 512
    # sing_val2 = 1/sing_val[: eigv_num]
    # rxi = (right_sing[: eigv_num, :].T * sing_val2) @ left_sing[:, : eigv_num].T @ whitened_measurement
    # rxi_alt = np.linalg.lstsq(mt_op, measurement, rcond=None)[0]
    # print(np.linalg.norm(rxi - rxi_alt), np.linalg.norm(rxi))

    # rxi_cov_op = mt_op.T.dot(mt_op)*noise_var
    # eigval, eigvec = np.linalg.eigh(rxi_cov_op)
    # print(eigval.shape, eigvec.shape)
    # tmp = eigvec.T[: eigv_num, :].dot(rxi)
    # print(tmp.shape)
    # rxi = eigvec[:, : eigv_num].dot(tmp)

    # omega = np.linalg.norm(src_img)**(-2) # ~1.3e-3 for the image of two slits
    # print(omega)
    # rxi = synth(measurement, mt_op, noise_var, omega)

    alpha_values = {("tc2", 3, 1e-1): 6e-3, ("tva2", 3, 1e-1): 0.158,
                    ("tc2", 2, 1e-1): 1e-3, ("tva2", 2, 1e-1): 0.158,
                    ("tc2", 6, 1e-1): 6e-3,
                    ("tc2", 7, 1e-1): 1e-3}

    alpha_values = defaultdict(lambda: 1e-5, alpha_values)
    # None, corresponding to alpha = 0+ would be a more accurate default value,
    # especially for no noise,
    # but this provides the same results faster.

    estimates = {}

    for processing_method, proc_method_name in zip(
            [compressive_l1,
             # compressive_l1_haar,
             compressive_tc2, compressive_tv_alt, compressive_tv_alt2],
            ["l1",
             # "l1h",
             "tc2", "tva", "tva2"]
    ):
        estimates[proc_method_name] = processing_method(
            measurement, mt_op, src_img.shape,
            alpha=alpha_values[(proc_method_name, img_id, float(noise_var))]
        )

    tau_values = {(2, 0.0): 1.0, (2, 0.1): 1,
                  (3, 0.): 1e-05, (3, 0.1): 0.1,
                  (6, 0.): 1.0, (6, 0.1): 1,
                  (7, 0.): 10.0, (7, 0.1): 1.,
                  (8, 0.): 1e-05, (8, 0.1): 1e-5}

    estimate_red_dense = dense_reduction(measurement, mt_op, src_img.shape)
    estimate_red_sparse = sparse_reduction(measurement, mt_op, src_img.shape,
                                           tau_values[(img_id, noise_var)], basis="eig")

    cs_part_names = ["l1",
                  # "l1h",
                  "tc2", "tva", "tva2"]
    if save:
        save_image_for_show(src_img, figure_name_format(img_id, noise_var, "src",
                                                        "", pattern_type=pattern_type))
        save_image_for_show(traditional_gi, figure_name_format(img_id, noise_var, "gi",
                                                        "", pattern_type=pattern_type))
        for cs_part_name in cs_part_names:
            save_image_for_show(
                estimates[cs_part_name], figure_name_format(
                    img_id, noise_var, cs_part_name,
                    alpha=alpha_values[(cs_part_name, img_id, float(noise_var))],
                    pattern_type=pattern_type
                )
            )
        save_image_for_show(estimate_red_dense, figure_name_format(img_id, noise_var, "red",
                                                        "", pattern_type=pattern_type))
        save_image_for_show(estimate_red_sparse, figure_name_format(
            img_id, noise_var, "reds",
            tau_values[(img_id, float(noise_var))], pattern_type=pattern_type
        ))

    if show:
        subplot_no = 0

        def plot_part(image, part_title):
            nonlocal subplot_no
            subplot_no += 1
            plt.subplot(3, 3, subplot_no)
            plt.imshow(image, cmap=plt.cm.gray, extent=[-size, size, -size, size]) #pylint: disable=E1101
            plt.xlabel("x, мкм")
            plt.ylabel("y, мкм")
            plt.title(part_title)

        fig = plt.gcf()
        # fig.clear()
        fig.set_tight_layout(True)

        plot_part(src_img, "Объект исследования")
        plot_part(traditional_gi, "Обычное ФИ")
        for name, desc in zip(cs_part_names,
                              ["нормы L1 в базисе DCT",
                               # "нормы L1 в базисе преобразования Хаара",
                               "полной кривизны",
                               "анизотропного вар-та вариации",
                               "альт. анизотропного вар-та вариации"]):
            plot_part(estimates[name],
                      "Сжатые измерения, минимизация " + desc)
        plot_part(estimate_red_dense,
                  "Редукция измерений без дополнительной информации об объекте")
        plot_part(estimate_red_sparse,
                  "Редукция измерений при дополнительной информации об объекте")
    # mng = plt.get_current_fig_manager()
    # try:
    #     mng.frame.Maximize(True)
    # except AttributeError:
    #     mng.window.showMaximized()
    # img_names = {2: "phys", 3: "two_slits", 6: "teapot128", 7: "teapot64"}
    # plt.savefig("../figures/{}_{}.pdf".format(
    #     img_names[img_id], "noisy" if noise_var > 0 else "noiseless"
    # ))
        plt.show()

    # for omega in [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]:
    #     estimate = synth(measurement, mt_op, src_img.shape, 1., omega)
    #     save_image_for_show(estimate, "synth_{:.0e}".format(omega), rescale=True)


def show_single_method(img_id=3, noise_var=0., n_measurements=1024, pattern_type: str="pseudorandom"):
    mt_op, illum_patterns, measurement, src_img, size = prepare_measurements(
        img_id=img_id, noise_var=noise_var, n_patterns=n_measurements,
        pattern_type=pattern_type
    )

    # "dct", no noise: 1e-5
    # "dct", 1e-2 noise: at least 1

    # basis = "eig"
    # # thr_coeff_values = [1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 1.]
    # # thr_coeff_values = [10, 100, 1e3, 1e4, 1e5]
    # thr_coeff_values = [5e-4, 5e-3]
    # for thr_coeff in thr_coeff_values:
    #     result = sparse_reduction(measurement, mt_op, src_img.shape,
    #                           thresholding_coeff=thr_coeff, basis=basis)
    #     diff_sq = np.linalg.norm(result - src_img)**2
    #     save_image_for_show(result.clip(0, None), "red_sparse_{}_{:.0e}_{}_{:.0e}".format(
    #         img_id, noise_var, basis, thr_coeff
    #     ), rescale=True)
    #     with open("red_sparse_diff.txt", "a", encoding="utf-8") as f:
    #         f.write("{}\t{:.1g}\t{}\t{:.1g}\t{:.3g}\n".format(img_id, noise_var, basis, thr_coeff, diff_sq))

    # result = compressive_tv_alt(measurement, mt_op, src_img.shape, alpha=1e-6)
    # # print(np.linalg.norm(result - src_img)**2)
    # plt.imshow(result, cmap=plt.cm.gray)
    # plt.show()

    # result = sparse_reduction(measurement, mt_op, src_img.shape,
    #                           thresholding_coeff=0.01, basis="eig")
    result = sparse_reduction(measurement, mt_op, src_img.shape,
                               thresholding_coeff=0.1, basis="eig")
    # src_img = load_demo_image(img_id, pad_by=32)
    # print(np.linalg.norm(result - src_img)**2)

    # result = np.linalg.lstsq(mt_op, measurement, rcond=None)[0].reshape(src_img.shape)

    plt.imshow(result, cmap=plt.cm.gray)

    plt.show()


def se_calculations(img_id=3, noise_var=0.1, tau_value=0.1, pattern_type: str="pseudorandom"):

    output = "../data/se_{}_{}_{:.0e}_{:.0e}.dat".format(img_id, pattern_type[0], noise_var, tau_value)
    intermediate_results = "../data/_se_{}_{}_{:.0e}_{:.0e}.dat".format(img_id, pattern_type[0], noise_var, tau_value)
    max_n_patterns = 13000
    total_mt_op, total_illum_patterns, total_measurement, src_img, _ = prepare_measurements(
        img_id=img_id, noise_var=noise_var, n_patterns=max_n_patterns,
        pattern_type=pattern_type
    )

    se_results = []
    header = ["n", "gi",
              # "l1",
              "tva", "tva2", "reds"]

    t_start = perf_counter()

    if pattern_type == "quasirandom":
        n_patterns_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 288, 320, 352, 368,
                             384, 400, 416, 448, 480, 512, 1024, 2048]
    else:
        n_patterns_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 288, 320, 352, 368,
                             384, 400, 416, 448, 480, 512, 1024, 2048, 4096, 8192,
                             max_n_patterns]
    se_results = np.empty((len(n_patterns_values), len(header)))
    header = '\t'.join(header)
    try:
        prev_results = np.loadtxt(intermediate_results)
    except OSError:
        start_ind = 0
    else:
        start_ind = prev_results.shape[0]
        se_results[: start_ind, :] = prev_results
        n_patterns_values = n_patterns_values[start_ind: ]

    for i, n_patterns in tqdm(enumerate(n_patterns_values, start_ind),
                              total=len(n_patterns_values)):
        measurement = total_measurement[: n_patterns]
        mt_op = total_mt_op[: n_patterns, ...]
        illum_patterns = total_illum_patterns[: n_patterns, ...]
        se_results[i, 0] = n_patterns

        # Traditional ghost imaging
        traditional_gi = np.tensordot(measurement - measurement.mean(),
                                      illum_patterns - illum_patterns.mean(axis=0),
                                      axes=1)/measurement.size
        se_results[i, 1] = np.linalg.norm(traditional_gi - src_img)**2

        # L1 compressive sensing
        # cs_l1 = compressive_l1(measurement, mt_op, src_img.shape, alpha=1e-5)
        # current_results.append(np.linalg.norm(cs_l1 - src_img)**2)

        # anisotropic TV compressive sensing
        cs_tva = compressive_tv_alt(measurement, mt_op, src_img.shape, alpha=1e-5)
        se_results[i, 2] = np.linalg.norm(cs_tva - src_img)**2

        # anisotropic TV compressive sensing, second version
        cs_tva2 = compressive_tv_alt2(measurement, mt_op, src_img.shape, alpha=1e-5)
        se_results[i, 3] = np.linalg.norm(cs_tva2 - src_img)**2

        # measurement reduction
        reds = sparse_reduction(measurement, mt_op, src_img.shape,
                                thresholding_coeff=tau_value)
        se_results[i, 4] = np.linalg.norm(reds - src_img)**2

        np.savetxt(intermediate_results, se_results[: i + 1, ...],
                   delimiter='\t', fmt='%.5g', header=header)

    t_end = perf_counter()
    logger.info("Calculations took {:.3g} s.".format(t_end - t_start))

    np.savetxt(output.format(img_id, noise_var), se_results,
               delimiter='\t', fmt='%.5g', header=header)
    try:
        remove(intermediate_results)
    except FileNotFoundError:
        pass
if __name__ == "__main__":
    # show_single_method(3, 0)
    # show_single_method(3, 1e-1)
    se_calculations(7, 1e-1, 1., pattern_type="quasirandom")
    se_calculations(7, 1e-1, 10., pattern_type="quasirandom")
    # show_single_method(3, 1e-1, 1024)
    # show_single_method(6, 0)
    # show_single_method(6, 1e-1)
    # show_single_method(7, 0)
    # show_single_method(7, 1e-1)
    # show_methods(8, 1e-1, n_patterns=512, save=True, show=True, pattern_type="speckle")
    # show_methods(3)
    # show_methods(3, 1e-1)
    # show_methods(2)
    # show_methods(2, 1e-1)
    # show_methods(6)
    # show_methods(6, 1e-1)
    # show_methods(7)
    # show_methods(7, 1e-1)
    # finding_alpha(3, 0., "l1")
    # finding_alpha(3, 0., "l1h")
    # finding_alpha(3, 0., "tc2")
    # finding_alpha(3, 0., "tva")
    # finding_alpha(3, 1e-1, "l1")
    # finding_alpha(3, 1e-1, "l1h")
    # finding_alpha(3, 1e-1, "tc2")
    # finding_alpha(3, 1e-1, "tva")
    # finding_alpha(2, 0., "l1")
    # finding_alpha(2, 0., "l1h")
    # finding_alpha(2, 0., "tc2")
    # finding_alpha(2, 0., "tva")
    # finding_alpha(2, 1e-1, "l1")
    # finding_alpha(2, 1e-1, "l1h")
    # finding_alpha(2, 1e-1, "tc2")
    # finding_alpha(2, 1e-1, "tva")
    # finding_alpha(6, 0., "l1")
    # finding_alpha(6, 0., "l1h")
    # finding_alpha(6, 0., "tc2")
    # finding_alpha(6, 0., "tva")
    # finding_alpha(6, 1e-1, "l1")
    # finding_alpha(6, 1e-1, "l1h")
    # finding_alpha(6, 1e-1, "tc2")
    # finding_alpha(6, 1e-1, "tva")
    # finding_alpha(7, 0., "tc2")
    # finding_alpha(7, 0., "tva")
    # finding_alpha(7, 1e-1, "tc2")
    # finding_alpha(7, 1e-1, "tva")
    # finding_alpha(7, 0., "l1")
    # finding_alpha(7, 0., "l1h")
    # finding_alpha(7, 1e-1, "l1")
    # finding_alpha(7, 1e-1, "l1h")
    # finding_alpha_l_curve(3, 1e-1, "l1")
    # finding_alpha_l_curve(3, 1e-1, "tva")
    # finding_alpha_l_curve(6, 1e-1, "l1")
    # finding_alpha_l_curve(6, 1e-1, "l1h")
    # finding_alpha_l_curve(6, 1e-1, "tc2")
    # finding_alpha_l_curve(6, 1e-1, "tva")
    # finding_alpha_l_curve(6, 1e-1, "tva2")
    # finding_alpha_l_curve(7, 1e-1, "l1")
    # finding_alpha_l_curve(7, 1e-1, "l1h")
    # finding_alpha_l_curve(7, 1e-1, "tc2")
    # finding_alpha_l_curve(7, 1e-1, "tva")
    # finding_alpha_l_curve(7, 1e-1, "tva2")
    # plot_l_curve(3, 1e-1, "l1")
    # finding_iter_params(3, 0.)
