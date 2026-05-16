"""Tests for physical constants -- values must match Fortran original."""

import math
import pytest
from ibis.constants import PhysicalConstants, get_constants


class TestPhysicalConstants:
    """Verify all constants match ibis_2.6b4/initial.f values."""

    def setup_method(self):
        self.c = get_constants()

    def test_stef(self):
        assert self.c.stef == pytest.approx(5.67e-8, rel=1e-6)

    def test_vonk(self):
        assert self.c.vonk == pytest.approx(0.4, rel=1e-6)

    def test_grav(self):
        assert self.c.grav == pytest.approx(9.80616, rel=1e-6)

    def test_tmelt(self):
        assert self.c.tmelt == pytest.approx(273.16, rel=1e-6)

    def test_hvap(self):
        assert self.c.hvap == pytest.approx(2.5104e6, rel=1e-6)

    def test_hfus(self):
        assert self.c.hfus == pytest.approx(0.3336e6, rel=1e-6)

    def test_hsub_derived(self):
        """hsub must equal hvap + hfus."""
        assert self.c.hsub == pytest.approx(self.c.hvap + self.c.hfus, rel=1e-10)

    def test_ch2o(self):
        assert self.c.ch2o == pytest.approx(4.218e3, rel=1e-6)

    def test_cice(self):
        assert self.c.cice == pytest.approx(2.106e3, rel=1e-6)

    def test_cair(self):
        assert self.c.cair == pytest.approx(1.00464e3, rel=1e-6)

    def test_cvap(self):
        assert self.c.cvap == pytest.approx(1.81e3, rel=1e-6)

    def test_rair(self):
        assert self.c.rair == pytest.approx(287.04, rel=1e-6)

    def test_rvap(self):
        assert self.c.rvap == pytest.approx(461.0, rel=1e-6)

    def test_cappa_derived(self):
        """cappa must equal rair / cair."""
        assert self.c.cappa == pytest.approx(self.c.rair / self.c.cair, rel=1e-10)

    def test_rhow(self):
        assert self.c.rhow == pytest.approx(1.0e3, rel=1e-6)

    def test_epsilon(self):
        assert self.c.epsilon == pytest.approx(1.0e-7, rel=1e-6)

    def test_pi(self):
        """Pi must match the F77 value (3.1415927), NOT high-precision."""
        assert self.c.pi == pytest.approx(3.1415927, rel=1e-7)

    def test_ndaypm(self):
        assert self.c.ndaypm == (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
        assert sum(self.c.ndaypm) == 365

    def test_is_frozen_dataclass(self):
        """PhysicalConstants should be immutable (equinox.Module)."""
        with pytest.raises(AttributeError):
            self.c.stef = 999.0
