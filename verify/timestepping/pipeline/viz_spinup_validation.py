"""CLM/ELM-style validation PDF for a spin-up year: input surface data + 7 output diagnostic pages.

Usage:
    python -m verify.timestepping.pipeline.viz_spinup_validation \\
        --year 200 \\
        --monthly-dir /shared/nas2/Hangkai/ibis/verify_outputs/spinup_200yr_0p5deg_v4_n4_carbon_monthly/monthly \\
        --out /home/hangkai/spinup_v4_year200_validation_with_inputs.pdf
"""
import argparse, os, sys, numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from netCDF4 import Dataset
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

PFT_NAMES_SHORT = ['TropEverGr','TropDecid','WarmTempEv','TempConifer','TempDecid',
                   'BorealConif','BorealDecid','BorealColdD','EvShrubs','DecShrubs',
                   'C4grass','C3grass']
RF_BIOMES = {
    1: 'TropEverGr', 2: 'TropDecid', 3: 'WTempBLEv', 4: 'TempCEv', 5: 'TempDecid',
    6: 'BorealEv', 7: 'BorealDec', 8: 'MixedEvDec', 9: 'Savanna', 10: 'Grassland',
    11: 'DenseShrub', 12: 'OpenShrub', 13: 'Tundra', 14: 'Desert', 15: 'PolarDesert'
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--monthly-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rf-vegtype",
                   default="/shared/nas2/Hangkai/ibis/inputs/ramankutty_foley_1999/vegtype_0.5.nc")
    args = p.parse_args()
    YEAR = args.year

    def load(m):
        ds = Dataset(f"{args.monthly_dir}/ibis_monthly_y{YEAR:04d}_m{m:02d}.nc")
        return {v: np.asarray(ds.variables[v][:]) for v in ds.variables if v != 'pft_name'}

    jan, jul, dec = load(1), load(7), load(12)
    lats, lons = dec["lat"], dec["lon"]
    print(f"Loaded {len(lats)} cells for year {YEAR}.")

    nan_cells = np.where(np.isnan(dec["tu"]))[0]
    print(f"  NaN cells in Y{YEAR}: {len(nan_cells)}  (will appear as missing pixel)")
    for ci in nan_cells:
        print(f"    cell {ci}: lat={lats[ci]:.2f}, lon={lons[ci]:.2f}")

    def to_grid(arr):
        g = np.full((360, 720), np.nan)
        lat_idx = ((89.75 - lats) / 0.5 + 0.5).astype(int).clip(0, 359)
        lon_idx = ((lons + 180.0) / 0.5 - 0.5).astype(int).clip(0, 719)
        g[lat_idx, lon_idx] = arr
        return g

    print("Loading HWSD soil + R&F vegetation IC...")
    from verify.timestepping.pipeline.global_gswp3_test import build_global_cell_grid
    from verify.timestepping.pipeline.real_state_builder import make_real_state_for_cells
    cells = build_global_cell_grid(lat_step=0.5, lon_step=0.5)
    soil_i, snow_i, atm_i, veg_i = make_real_state_for_cells(cells, vegtype_source="ramankutty")
    rf = Dataset(args.rf_vegtype)
    rf_vegtype_grid = np.asarray(rf.variables["vegtype"][0, 0, :, :])

    with PdfPages(args.out) as pdf:
        extent = [-180, 180, -90, 90]

        # ============== INPUT PAGE A: HWSD soil ==============
        fig, axs = plt.subplots(3, 2, figsize=(16, 14))
        sand = np.asarray(soil_i["sand"]); clay = np.asarray(soil_i["clay"])
        poros = np.asarray(soil_i["poros"]); hydraul = np.asarray(soil_i["hydraul"])
        for ax, data, title, vmin, vmax, cmap in [
            (axs[0,0], sand[:, 0], "INPUT: HWSD sand fraction, layer 0 (surface)", 0, 100, 'YlOrBr'),
            (axs[0,1], clay[:, 0], "INPUT: HWSD clay fraction, layer 0 (surface)", 0, 70, 'PuRd'),
            (axs[1,0], sand[:, 5], "INPUT: HWSD sand fraction, layer 5 (deepest)", 0, 100, 'YlOrBr'),
            (axs[1,1], clay[:, 5], "INPUT: HWSD clay fraction, layer 5 (deepest)", 0, 70, 'PuRd'),
            (axs[2,0], poros[:, 0], "INPUT: Soil porosity (layer 0)", 0.3, 0.6, 'YlGnBu'),
            (axs[2,1], np.log10(np.maximum(hydraul[:, 0], 1e-10)),
                "INPUT: log10(hydraulic conductivity m/s) layer 0", -8, -4, 'Blues'),
        ]:
            g = to_grid(data)
            im = ax.imshow(g, extent=extent, origin='upper', cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
            ax.set_title(title); ax.set_xlabel('lon'); ax.set_ylabel('lat')
            plt.colorbar(im, ax=ax, shrink=0.7)
        fig.suptitle('INPUTS: HWSD-derived soil hydraulic & textural properties', fontsize=14, y=1.0)
        plt.tight_layout(); pdf.savefig(fig); plt.close()

        # ============== INPUT PAGE B: R&F vegetation + IC ==============
        fig, axs = plt.subplots(3, 2, figsize=(16, 14))
        im = axs[0,0].imshow(rf_vegtype_grid, extent=extent, origin='upper', cmap=plt.cm.tab20,
                              vmin=0.5, vmax=15.5, aspect='auto')
        axs[0,0].set_title("INPUT: Ramankutty & Foley 1999 vegetation type (15 biomes)")
        axs[0,0].set_xlabel('lon'); axs[0,0].set_ylabel('lat')
        cbar = plt.colorbar(im, ax=axs[0,0], shrink=0.7, ticks=range(1, 16))
        cbar.set_ticklabels([f'{k}:{v[:8]}' for k, v in RF_BIOMES.items()])

        fu_ic = np.asarray(veg_i["fu"]); fl_ic = np.asarray(veg_i["fl"])
        lai_ic = np.asarray(veg_i["lai"])
        biomass_ic = np.asarray(veg_i["biomass"]).sum(axis=1)
        for ax, data, title, vmin, vmax, cmap in [
            (axs[0,1], biomass_ic, "INPUT: IC total biomass (kgC/m²) from R&F + initialize_vegetation", 0, 30, 'YlGn'),
            (axs[1,0], fu_ic, "INPUT: IC upper canopy cover (fu)", 0, 1, 'Greens'),
            (axs[1,1], fl_ic, "INPUT: IC lower canopy cover (fl)", 0, 1, 'YlGn'),
            (axs[2,0], lai_ic[:, 1], "INPUT: IC upper canopy LAI", 0, 7, 'Greens'),
            (axs[2,1], lai_ic[:, 0], "INPUT: IC lower canopy LAI", 0, 5, 'YlGn'),
        ]:
            g = to_grid(data)
            im = ax.imshow(g, extent=extent, origin='upper', cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
            ax.set_title(title); ax.set_xlabel('lon'); ax.set_ylabel('lat')
            plt.colorbar(im, ax=ax, shrink=0.7)
        fig.suptitle('INPUTS: Ramankutty & Foley 1999 vegetation + derived IC', fontsize=14, y=1.0)
        plt.tight_layout(); pdf.savefig(fig); plt.close()

        # ============== INPUT PAGE C: IC per-PFT biomass ==============
        cbiol_ic = np.asarray(veg_i["cbiol"]); cbiow_ic = np.asarray(veg_i["cbiow"])
        fig, axs = plt.subplots(3, 2, figsize=(16, 14))
        for ax, pft_i in zip(axs.flat, [0, 4, 5, 6, 10, 11]):
            biomass_pft = cbiol_ic[:, pft_i] + cbiow_ic[:, pft_i] + np.asarray(veg_i["cbior"])[:, pft_i]
            g = to_grid(biomass_pft)
            vmax = 20 if pft_i < 8 else 3
            im = ax.imshow(g, extent=extent, origin='upper', cmap='YlGn', vmin=0, vmax=vmax, aspect='auto')
            ax.set_title(f'INPUT IC: PFT {pft_i+1} ({PFT_NAMES_SHORT[pft_i]}) biomass (kgC/m²)')
            plt.colorbar(im, ax=ax, shrink=0.7)
        fig.suptitle('INPUTS: IC per-PFT biomass distribution (cold-start, exist=1 globally)', fontsize=14, y=1.0)
        plt.tight_layout(); pdf.savefig(fig); plt.close()

        # ============== OUTPUT PAGE 1: Annual carbon cycle ==============
        fig, axs = plt.subplots(3, 2, figsize=(16, 14))
        for ax, key, title, vmin, vmax, cmap in [
            (axs[0,0], "GPP",  f"OUTPUT Y{YEAR} annual GPP (kgC/m²/yr)",       0,    5, 'YlGn'),
            (axs[0,1], "NPP",  f"OUTPUT Y{YEAR} annual NPP (kgC/m²/yr)",      -1,    2, 'RdYlGn'),
            (axs[1,0], "NEE",  f"OUTPUT Y{YEAR} annual NEE (neg=C sink)",     -2,    2, 'RdBu_r'),
            (axs[1,1], "biomass_total", f"OUTPUT Y{YEAR} dec: biomass total (kgC/m²)", 0, 30, 'YlGn'),
            (axs[2,0], "LAI_upper", f"OUTPUT Y{YEAR} dec: upper canopy LAI",   0,    7, 'Greens'),
            (axs[2,1], "LAI_lower", f"OUTPUT Y{YEAR} dec: lower canopy LAI",   0,    5, 'YlGn'),
        ]:
            g = to_grid(dec[key])
            im = ax.imshow(g, extent=extent, origin='upper', cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
            ax.set_title(title); ax.set_xlabel('lon'); ax.set_ylabel('lat')
            plt.colorbar(im, ax=ax, shrink=0.7)
        fig.suptitle(f'OUTPUTS YEAR {YEAR}: Carbon cycle', fontsize=14, y=1.0)
        plt.tight_layout(); pdf.savefig(fig); plt.close()

        # ============== OUTPUT PAGE 2: Per-PFT biomass ==============
        fig, axs = plt.subplots(3, 2, figsize=(16, 14))
        for ax, pft_i in zip(axs.flat, [0, 4, 5, 6, 10, 11]):
            g = to_grid(dec["biomass_pft"][:, pft_i])
            vmax = 30 if pft_i < 8 else 5
            im = ax.imshow(g, extent=extent, origin='upper', cmap='YlGn', vmin=0, vmax=vmax, aspect='auto')
            ax.set_title(f'OUTPUT Y{YEAR}: PFT {pft_i+1} ({PFT_NAMES_SHORT[pft_i]}) biomass (kgC/m²)')
            plt.colorbar(im, ax=ax, shrink=0.7)
        fig.suptitle(f'OUTPUTS Y{YEAR}: per-PFT biomass (compare to IC)', fontsize=14, y=1.0)
        plt.tight_layout(); pdf.savefig(fig); plt.close()

        # ============== OUTPUT PAGE 3: Per-PFT annual GPP ==============
        fig, axs = plt.subplots(3, 2, figsize=(16, 14))
        for ax, pft_i in zip(axs.flat, [0, 4, 5, 6, 10, 11]):
            g = to_grid(dec["GPP_pft"][:, pft_i])
            im = ax.imshow(g, extent=extent, origin='upper', cmap='YlGn', vmin=0, vmax=3, aspect='auto')
            ax.set_title(f'OUTPUT Y{YEAR}: PFT {pft_i+1} ({PFT_NAMES_SHORT[pft_i]}) annual GPP (kgC/m²/yr)')
            plt.colorbar(im, ax=ax, shrink=0.7)
        plt.tight_layout(); pdf.savefig(fig); plt.close()

        # ============== OUTPUT PAGE 4: Seasonal contrast ==============
        fig, axs = plt.subplots(3, 2, figsize=(16, 14))
        for ax, key, fd, title, vmin, vmax, cmap, off in [
            (axs[0,0], "GPP", jan, f"Y{YEAR} Jan cum GPP (kgC/m²)", 0, 0.5, 'YlGn', 0),
            (axs[0,1], "GPP", jul, f"Y{YEAR} Jul cum GPP (kgC/m²)", 0, 3, 'YlGn', 0),
            (axs[1,0], "tg",  jan, f"Y{YEAR} Jan ground T (°C)", -50, 35, 'RdYlBu_r', -273.15),
            (axs[1,1], "tg",  jul, f"Y{YEAR} Jul ground T (°C)", -30, 40, 'RdYlBu_r', -273.15),
            (axs[2,0], "snow_fi", jan, f"Y{YEAR} Jan snow cover", 0, 1, 'Blues', 0),
            (axs[2,1], "snow_fi", jul, f"Y{YEAR} Jul snow cover", 0, 1, 'Blues', 0),
        ]:
            data = fd[key] + off
            g = to_grid(data)
            im = ax.imshow(g, extent=extent, origin='upper', cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
            ax.set_title(title); ax.set_xlabel('lon')
            plt.colorbar(im, ax=ax, shrink=0.7)
        fig.suptitle(f'OUTPUTS Y{YEAR}: Seasonal contrast', fontsize=14, y=1.0)
        plt.tight_layout(); pdf.savefig(fig); plt.close()

        # ============== OUTPUT PAGE 5: Surface state ==============
        fig, axs = plt.subplots(3, 2, figsize=(16, 14))
        for ax, key, title, vmin, vmax, cmap, off in [
            (axs[0,0], "tu", f"Y{YEAR} Dec upper canopy T (°C)", -40, 35, 'RdYlBu_r', -273.15),
            (axs[0,1], "tl", f"Y{YEAR} Dec lower canopy T (°C)", -40, 35, 'RdYlBu_r', -273.15),
            (axs[1,0], "wsoi", f"Y{YEAR} Dec surface soil moisture", 0, 1, 'YlGnBu', 0),
            (axs[1,1], "wisoi", f"Y{YEAR} Dec surface soil ice", 0, 1, 'PuBu', 0),
            (axs[2,0], "fu",  f"Y{YEAR} Dec upper canopy cover (fu)", 0, 1, 'Greens', 0),
            (axs[2,1], "fl",  f"Y{YEAR} Dec lower canopy cover (fl)", 0, 1, 'YlGn', 0),
        ]:
            data = dec[key]
            if data.ndim == 2: data = data[:, 0]
            data = data + off
            g = to_grid(data)
            im = ax.imshow(g, extent=extent, origin='upper', cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
            ax.set_title(title)
            plt.colorbar(im, ax=ax, shrink=0.7)
        plt.tight_layout(); pdf.savefig(fig); plt.close()

        # ============== OUTPUT PAGE 6: Water fluxes ==============
        fig, axs = plt.subplots(3, 2, figsize=(16, 14))
        for ax, key, title, vmin, vmax, cmap, scale in [
            (axs[0,0], "gtrans", "Jul transpiration (mm/day)", 0, 8, 'YlGnBu', 86400),
            (axs[0,1], "gsuvap", "Jul soil evap (mm/day)",     0, 5, 'YlGnBu', 86400),
            (axs[1,0], "ginvap", "Jul canopy intercept evap (mm/day)", 0, 3, 'YlGnBu', 86400),
            (axs[1,1], "grunof", "Jul surface runoff (mm/day)", 0, 5, 'Blues', 86400),
            (axs[2,0], "gdrain", "Jul drainage (mm/day)", 0, 5, 'Blues', 86400),
            (axs[2,1], "tg",     "Jul ground T (°C)", -30, 40, 'RdYlBu_r', None),
        ]:
            data = jul[key]
            data = data * scale if scale else data - 273.15
            g = to_grid(data)
            im = ax.imshow(g, extent=extent, origin='upper', cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
            ax.set_title(f'Y{YEAR} ' + title)
            plt.colorbar(im, ax=ax, shrink=0.7)
        plt.tight_layout(); pdf.savefig(fig); plt.close()

        # ============== OUTPUT PAGE 7: Soil profiles at named cells ==============
        LOCS = [
            ("Manaus (Amazon)", -3.0, -60.0), ("Yakutsk (Siberia)", 62.0, 130.0),
            ("Cairo (Sahara)", 30.0, 31.0),   ("Mumbai (Monsoon)", 19.0, 73.0),
            ("Madison (US Midwest)", 43.0, -89.0), ("Antarctic interior", -78.0, 107.0),
            ("Sydney (S Hemi)", -34.0, 151.0),
            ("Hudson Bay BUG cell (Y192 NaN)", 61.5, -94.5),
        ]
        fig, axs = plt.subplots(4, 2, figsize=(16, 14))
        soil_depths = np.array([0.05, 0.15, 0.35, 0.75, 1.5, 3.0])
        for ax_i, (name, qlat, qlon) in enumerate(LOCS):
            ax = axs[ax_i // 2, ax_i % 2]
            i = int(((lats - qlat)**2 + (lons - qlon)**2).argmin())
            ax.plot(jul["tsoi"][i] - 273.15, -soil_depths, 'ro-', label='Jul tsoi')
            ax.plot(jan["tsoi"][i] - 273.15, -soil_depths, 'bo-', label='Jan tsoi')
            ax.axvline(0, c='gray', lw=0.5)
            ax2 = ax.twiny()
            ax2.plot(jul["wsoi"][i], -soil_depths, 'g^--', alpha=0.6, label='Jul wsoi')
            ax2.plot(jan["wsoi"][i], -soil_depths, 'c^--', alpha=0.6, label='Jan wsoi')
            ax2.set_xlim(0, 1)
            ax.set_title(f'{name} (lat={lats[i]:.1f}, lon={lons[i]:.1f})')
            ax.set_xlabel('Soil T (°C)'); ax.set_ylabel('Depth (m)')
            ax2.set_xlabel('wsoi (fraction)', color='g')
            ax.legend(loc='lower right', fontsize=7)
            ax2.legend(loc='lower left', fontsize=7)
        plt.tight_layout(); pdf.savefig(fig); plt.close()

    print(f"\nWrote {args.out}  ({os.path.getsize(args.out)/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
