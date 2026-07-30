import torch

from torch_optics.doe import DOELayer


def test_orthogonal_rms_basis_has_nine_equal_scale_independent_modes():
    torch.manual_seed(123)
    doe = DOELayer(
        doe_type="New",
        trainable=True,
        basis_mode="orthogonal_rms",
        basis_rank=9,
        basis_rank_rtol=1e-4,
        basis_rms_m=3e-6,
        coeff_norm_limit=1.0,
        init_coeff_norm=1.0,
    )

    assert tuple(doe.zernike_basis.shape) == (9, 128, 128)
    assert doe.zernike_source_indices.tolist() == [0, 1, 2, 3, 4, 5, 6, 10, 11]
    torch.testing.assert_close(
        doe.zernike_coeffs.norm(),
        torch.tensor(1.0),
        atol=1e-6,
        rtol=0,
    )

    pupil = doe.spiral_p > 0.5
    pupil_modes = doe.zernike_basis[:, pupil]
    gram = pupil_modes @ pupil_modes.t() / pupil.sum()
    expected = torch.eye(9) * (3e-6 ** 2)
    torch.testing.assert_close(gram, expected, rtol=1e-5, atol=2e-18)
    torch.testing.assert_close(
        doe.pupil_rms(doe.heightmap()),
        torch.tensor(3e-6),
        rtol=1e-5,
        atol=0,
    )


def test_orthogonal_coefficients_start_feasible_and_clamp_to_physical_limit():
    torch.manual_seed(7)
    doe = DOELayer(
        doe_type="New",
        trainable=True,
        basis_mode="orthogonal_rms",
        basis_rank=9,
        basis_rms_m=3e-6,
        coeff_norm_limit=0.8,
        init_coeff_norm=0.6,
    )

    torch.testing.assert_close(
        doe.zernike_coeffs.norm(),
        torch.tensor(0.6),
        atol=1e-6,
        rtol=0,
    )
    with torch.no_grad():
        doe.zernike_coeffs.mul_(4.0)
    doe.clamp_parameters_()
    torch.testing.assert_close(
        doe.zernike_coeffs.norm(),
        torch.tensor(0.8),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        doe.pupil_rms(doe.heightmap()),
        torch.tensor(2.4e-6),
        rtol=1e-5,
        atol=0,
    )


def test_legacy_raw12_state_dict_remains_strictly_compatible():
    torch.manual_seed(11)
    original = DOELayer(
        doe_type="New",
        trainable=True,
        basis_mode="legacy_raw12",
    )
    restored = DOELayer(
        doe_type="New",
        trainable=True,
        basis_mode="legacy_raw12",
    )
    result = restored.load_state_dict(original.state_dict(), strict=True)

    assert tuple(restored.zernike_coeffs.shape) == (12,)
    assert not result.missing_keys
    assert not result.unexpected_keys
    assert torch.equal(restored.zernike_coeffs, original.zernike_coeffs)
