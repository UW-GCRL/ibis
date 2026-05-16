c ---------------------------------------------------------------------
      subroutine pheno
c ---------------------------------------------------------------------
c
c common blocks
c
      include 'implicit.h'
c
      include 'compar.h'
      include 'comatm.h'
      include 'comsoi.h'
      include 'comsum.h'
      include 'comveg.h'
c
c local variables
c
      integer
     >  i,          !
     >  imonth,     !
     >  iday        !
c
      real
     >  ddays,        !
     >  ddfac,        !
     >  tthreshold,   ! temperature threshold for budburst and senescence
     >  gthreshold,   ! temperature threshold for budburst and senescence
     >  avglaiu,      ! average lai of upper canopy 
     >  avglail,      ! average lai of lower canopy 
**** DTP 2000/06/28 Modified this following discussion with Navin. We now
*    retain fu(i) derived from dynaveg and constrain it to a local value
*    "fu_phys" in the range 0.25 to 0.975 used in the canopy physics calcs.
     >  fu_phys       ! Local value of fu(i) constrained to range 0.25 to 0.975
                      ! to keep physics calculations stable.
c
c define 'drop days' -- number of days to affect phenology change
c
      ddays = 15.0
      ddfac = 1.0 / ddays
c
c begin global grid
c
      do 100 i = 1, npoi
c
c ---------------------------------------------------------------------
c * * * upper canopy winter phenology * * *
c ---------------------------------------------------------------------
c
c temperature threshold for budburst and senescence
c
c temperature threshold is assumed to be 0 degrees C 
c or 5 degrees warmer than the coldest monthly temperature
c
        tthreshold = max (0.0         + 273.16,
     >                    tc(i) + 5.0 + 273.16)
c
c gdd threshold temperature for leaf budburst
c with a growing degree threshold of 100 units
c
        gthreshold = 0.0 + 273.16
c 
c determine if growing degree days are initiated
c
        if (a10td(i).lt.gthreshold) then
          agddu(i)  = 0.0
        else
          agddu(i) = agddu(i) + td(i) - gthreshold
        endif
c
c determine leaf display
c
        if (a10td(i).lt.tthreshold) then
          tempu(i)  = max (0.0, tempu(i) - ddfac)
        else
          tempu(i) = min (1., max (0.0, agddu(i) - 100.0) / 50.0)
        endif
c
c ---------------------------------------------------------------------
c * * * lower canopy winter phenology * * *
c ---------------------------------------------------------------------
c
c temperature threshold for budburst and senescence
c
c temperature threshold is assumed to be 0 degrees C 
c
        tthreshold = 0.0 + 273.16
c
c gdd threshold temperature for leaf budburst
c with a growing degree threshold of 150 units
c
        gthreshold = -5.0 + 273.16
c 
c determine if growing degree days are initiated
c
        if (a10td(i).lt.gthreshold) then
          agddl(i)  = 0.0
        else
          agddl(i) = agddl(i) + td(i) - gthreshold
        endif
c
c determine leaf display
c
        if (a10td(i).lt.tthreshold) then
          templ(i)  = max (0.0, templ(i) - ddfac)
        else
          templ(i) = min (1., max (0.0, agddl(i) - 150.0) / 50.0)
        endif
c
c ---------------------------------------------------------------------
c * * * drought canopy winter phenology * * *
c ---------------------------------------------------------------------
c
        if (a10ancub(i).lt.0.0) dropu(i) = max (0.1, dropu(i) - ddfac)
        if (a10ancub(i).ge.0.0) dropu(i) = min (1.0, dropu(i) + ddfac)
c
        if (a10ancls(i).lt.0.0) dropls(i) = max (0.1, dropls(i) - ddfac)
        if (a10ancls(i).ge.0.0) dropls(i) = min (1.0, dropls(i) + ddfac)
c
        if (a10ancl4(i).lt.0.0) dropl4(i) = max (0.1, dropl4(i) - ddfac)
        if (a10ancl4(i).ge.0.0) dropl4(i) = min (1.0, dropl4(i) + ddfac)
c
        if (a10ancl3(i).lt.0.0) dropl3(i) = max (0.1, dropl3(i) - ddfac)
        if (a10ancl3(i).ge.0.0) dropl3(i) = min (1.0, dropl3(i) + ddfac)
c
c ---------------------------------------------------------------------
c * * * update lai and canopy fractions * * *
c ---------------------------------------------------------------------
c
c upper canopy single sided leaf area index (area-weighted)
c
        avglaiu = plai(i,1)             +
     >            plai(i,2) * dropu(i)  +
     >            plai(i,3)             +
     >            plai(i,4)             +
     >            plai(i,5) * tempu(i)  +
     >            plai(i,6)             +
     >            plai(i,7) * tempu(i)  +
     >            plai(i,8) * tempu(i)
c
c upper canopy fractions
c
        frac(i,1) = plai(i,1)            / max (avglaiu, epsilon)
        frac(i,2) = plai(i,2) * dropu(i) / max (avglaiu, epsilon)
        frac(i,3) = plai(i,3)            / max (avglaiu, epsilon)
        frac(i,4) = plai(i,4)            / max (avglaiu, epsilon)
        frac(i,5) = plai(i,5) * tempu(i) / max (avglaiu, epsilon)
        frac(i,6) = plai(i,6)            / max (avglaiu, epsilon)
        frac(i,7) = plai(i,7) * tempu(i) / max (avglaiu, epsilon)
        frac(i,8) = plai(i,8) * tempu(i) / max (avglaiu, epsilon)
c
c lower canopy single sided leaf area index (area-weighted)
c
        avglail = plai(i,9)                              +
     >            plai(i,10) * min (templ(i), dropls(i)) +
     >            plai(i,11) * min (templ(i), dropl4(i)) +
     >            plai(i,12) * min (templ(i), dropl3(i))
c
c lower canopy fractions
c
        frac(i,9)  = plai(i,9)                              /
     >               max (avglail, epsilon)
c
        frac(i,10) = plai(i,10) * min (templ(i), dropls(i)) /
     >               max (avglail, epsilon)
c
        frac(i,11) = plai(i,11) * min (templ(i), dropl4(i)) /
     >               max (avglail, epsilon)
c
        frac(i,12) = plai(i,12) * min (templ(i), dropl3(i)) /
     >               max (avglail, epsilon)
c
c calculate the canopy leaf area index using the fractional vegetation cover
c
        lai(i,1) = avglail / fl(i)

**** DTP 2000/06/28 Modified this following discussion with Navin. We now
*    retain fu(i) derived from dynaveg and constrain it to a local value
*    "fu_phys" in the range 0.25 to 0.975 used in the canopy physics calcs.

        fu_phys = max (0.25, min (0.975, fu(i)))
        lai(i,2) = avglaiu / fu_phys
        lai(i,2) = avglaiu / fu(i)
c
c put a fix on canopy lais to avoid problems in physics
c
        lai(i,1) = min (lai(i,1), 12.0)
        lai(i,2) = min (lai(i,2), 12.0)
c
c ---------------------------------------------------------------------
c * * * update canopy height parameters * * *
c ---------------------------------------------------------------------
c
c update lower canopy height parameters
c
c note that they are based on vegetation fraction and not
c averaged over the entire gridcell
c
        zbot(i,1)   =  0.05
        ztop(i,1)   =  max (0.25, lai(i,1) * 0.25)
c        
c constrain ztop to be at least 0.5 meter lower than 
c zbot for upper canopy
c
        ztop(i,1) = min (ztop(i,1), zbot(i,2) - 0.5)
c
c end of loop
c
 100  continue
c
c return to main program
c 
      return
      end
c
