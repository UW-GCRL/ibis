      subroutine fwetcal
c ---------------------------------------------------------------------
c
c calculates fwet[u,s,l], the fractional areas wetted by 
c intercepted h2o (liquid and snow combined) -  the maximum value
c fmax (<1) allows some transpiration even in soaked conditions
c
c use a linear relation between fwet* and wliq*,wsno* (at least
c for small values), so that the implied "thickness" is constant
c (equal to wliq*max, wsno*max as below) and the typical amount
c evaporated in one timestep in steph2o will not make wliq*,wsno*
c negative and thus cause a spurious unrecoverable h2o loss
c
c (the max(w*max,.01) below numericaly allows w*max = 0 without
c blowup.) in fact evaporation in one timestep *does* sometimes
c exceed wliq*max (currently 1 kg/m2), so there is an additional
c safeguard in turvap that limits the wetted-area aerodynamic
c coefficients suw,ssw,slw -- if that too fails, there is an 
c ad-hoc adjustment in steph2o2 to reset negative wliq*,wsno*
c amounts to zero by taking some water vapor from the atmosphere.
c
c also sets rliq[u,s,l], the proportion of fwet[u,s,l] due to
c liquid alone. fwet,rliq are used in turvap, rliq in steph2o. 
c (so rliq*fwet, (1-rliq)*fwet are the fractional areas wetted
c by liquid and snow individually.) if fwet is 0, choose rliq
c = 1 if t[u,s,l] ge tmelt or 0 otherwize, for use by turvap and
c steph2o in case of initial dew formation on dry surface.
c 
      include 'implicit.h'
c
      include 'compar.h'
      include 'comveg.h'
      include 'com1d.h'
c
c local variables
c
      integer i           ! loop indice
c
      real fmax,          ! maximum water cover on two-sided leaf
     >     xliq,          ! fraction of wetted leaf (liquid only)
     >     xtot           ! fraction of wetted leaf (liquid and snow)
c
c maximum water cover on two-sided leaf
c
      parameter (fmax = 0.25)
c
c upper leaves
c
      do 100 i = 1, npoi
c
        xliq = wliqu(i) / max (wliqumax, 0.01)
        xtot = xliq + wsnou(i) / max (wsnoumax, 0.01)
c
        fwetu(i) = min (fmax, xtot)
        rliqu(i) = xliq / max (xtot, epsilon)
c
        if (fwetu(i).eq.0.0) then
          rliqu(i) = 1.0
          if (tu(i).lt.tmelt) rliqu(i) = 0.0
        endif
c
  100 continue
c
c upper stems
c
      do 200 i = 1, npoi
c
        xliq = wliqs(i) / max (wliqsmax, 0.01)
        xtot = xliq + wsnos(i) / max (wsnosmax, 0.01)
c
        fwets(i) = min (fmax, xtot)
        rliqs(i) = xliq / max (xtot, epsilon)
c
        if (fwets(i).eq.0.0) then
          rliqs(i) = 1.0
          if (ts(i).lt.tmelt) rliqs(i) = 0.0
        endif
c
  200 continue
c
c lower veg
c
      do 300 i = 1, npoi
c
        xliq = wliql(i) / max (wliqlmax, 0.01)
        xtot = xliq + wsnol(i) / max (wsnolmax, 0.01)
c
        fwetl(i) = min (fmax, xtot)
        rliql(i) = xliq / max (xtot, epsilon)
c
        if (fwetl(i).eq.0.) then
          rliql(i) = 1.0
          if (tl(i).lt.tmelt) rliql(i) = 0.0
        endif
c
  300 continue
c
      return
      end
c
c
      subroutine mix (xm,tm, x1,t1, x2,t2, x3,t3)
c ---------------------------------------------------------------------
c
c calorimetrically mixes masses x1,x2,x3 with temperatures
c t1,t2,t3 into combined mass xm with temperature tm
c
c xm,tm may be returned into same location as one of x1,t1,..,
c so hold result temporarily in xtmp,ttmp below
c
c will work if some of x1,x2,x3 have opposite signs, but may 
c give unphysical tm's
c
      include 'implicit.h'
c
      include 'compar.h'
c
c Arguments (input except for xm, tm)
c
      real xm(npoi),     ! resulting mass  
     >     tm(npoi),     ! resulting temp
     >     x1(npoi),     ! mass 1
     >     t1(npoi),     ! temp 1
     >     x2(npoi),     ! mass 2
     >     t2(npoi),     ! temp 2
     >     x3(npoi),     ! mass 3
     >     t3(npoi)      ! temp 3
c
c local variables
c
      integer i          ! loop indice
c
      real xtmp,         ! resulting mass (storing variable)
     >     ytmp,         !  "
     >     ttmp          ! resulting temp
c
c ---------------------------------------------------------------------
c
      do 100 i=1,npoi
c
        xtmp = x1(i) + x2(i) + x3(i)
c
        ytmp = sign (max (abs(xtmp), epsilon), xtmp)
c
        if (abs(xtmp).ge.epsilon) then
          ttmp = (t1(i)*x1(i) + t2(i)*x2(i) + t3(i)*x3(i)) / ytmp
        else
          ttmp = 0.
          xtmp = 0.
        endif
c
        xm(i) = xtmp
        tm(i) = ttmp
c
  100 continue
c
      return
      end
c
c
      subroutine noveg
c ---------------------------------------------------------------------
c
c if no veg surfaces exist, set prog vars to nominal values
c
c (sensible fluxes fsen[u,s,l], latent fluxes fvap[u,s,l]*, 
c temperature t[u,s,l], and intercepted liquid, snow amounts 
c wliq[u,s,l], wsno[u,s,l] have been calculated for a unit 
c leaf/stem surface, whether or not one exists.)
c
      include 'implicit.h'
c
      include 'compar.h'
      include 'comsno.h'
      include 'comsoi.h'
      include 'comveg.h'
c
c local variables
c
      integer i   ! loop indice
c
      real tav,   ! average temp for soil and snow 
     >     x,     ! total lai + sai
     >     y      ! fraction of lower canopy not snow covered 
c
      do 100 i = 1, npoi
c
        tav = (1.-fi(i))*tg(i) + fi(i)*ti(i)
c
        if (lai(i,2).eq.0. .or. fu(i).eq.0.) then
          tu(i) = tav
          wliqu(i) = 0.
          wsnou(i) = 0.
        endif
c
        if (sai(i,2).eq.0. .or. fu(i).eq.0.) then
          ts(i) = tav
          wliqs(i) = 0.
          wsnos(i) = 0.
        endif 
c
        x = 2.0 * (lai(i,1) + sai(i,1))
        y = fl(i)*(1.-fi(i))
c
        if (x .eq.0. .or. y.eq.0.) then
          tl(i) = tav 
          wliql(i) = 0.
          wsnol(i) = 0.
        endif
c
  100 continue
c
      return
      end
c
      real function twet3(tak, q, p)
c ------------------------------------------------------------------------
c
c twet3.f last update 8/30/2000 C Molling
c
c This function calculates the wet bulb temperature given
c air temp, specific humidity, and air pressure.  It needs the function esat
c in order to work (in comsat.h).  The function is an approximation to
c the actual wet bulb temperature relationship.  It agrees well with the
c formula in the Smithsonian Met. Tables for moderate humidities, but differs
c by as much as 1 K in extremely dry or moist environments.
c
c INPUT
c     tak - air temp in K
c     q - specific humidity in kg/kg
c     p - air pressure in Pa (Pa = 100 * mb)
c
c OUTPUT
c     twet3 - wet bulb temp in K, accuracy?
c
      include 'implicit.h'
      include 'compar.h'
c
      integer i
c
      real tak, q, p, ta, twk, twold, diff
c
      include 'comsat.h'
c
c temperatures in twet3 equation must be in C
c pressure in qsat function must be in Pa
c temperatures in esat,hvapf functions must be in K
c
c     Air temp in C
c     -------------
      ta = tak - 273.16
c
c     First guess for wet bulb temp in C, K
c     -------------------------------------
      twet3 = ta * q / qsat(esat(tak),p)
      twk = twet3 + 273.16
c
c     Iterate to converge
c     -------------------
      do 100 i = 1, 20
         twold = twk - 273.16
         twet3 = ta - (hvapf(twk,tak)/cair) * ( qsat( esat(twk),p )-q )
         diff = twet3 - twold
c
c below, the 0.2 is the relaxation parameter that works up to 40C (at least)
c
         twk = twold + 0.2 * diff + 273.16
         if (abs(twk-273.16-twold) .lt. 0.02) goto 999
 100  continue
c
      print *, 'Warning, twet3 failed to converge after 20 iterations!'
      print *, 'twet3, twetold: ', twk, twold+273.16
      print *, 'twetbulb is being set to the air temperature'
c
      twet3 = tak
c
c     Return wet bulb temperature in K
c     --------------------------------
 999  twet3 = twk
c
      return
      end
c scopy and const utility subroutines removed from this subset:
c they are not called by any carved canopy subroutine. If needed in the
c future, copy them from utilities.f.
c


      real function cvmgt (x, y, l)
      real x, y
      logical l
      if (l) then
        cvmgt = x
      else
        cvmgt = y
      endif
      return
      end
      subroutine steph2o
     >  (tveg,  wliq,  wsno,  xai,  pflux,  rain, train, snow, tsnow,
     >   tdrip, tblow, wliqmax, wsnomax, wliqmin, wsnomin)
c ---------------------------------------------------------------------
c
c steps intercepted h2o for one canopy component (upper leaves, 
c upper stems, or lower veg) through one lsx time step, adjusting
c for h2o sensible heat and phase changes. also modifies precip
c due to interception and drip,blowoff
c
c 
c
      include 'implicit.h'
c
      include 'compar.h'
c
c Arguments (all arguments are supplied (unchanged) unless otherwise noted
c
      real tdrip,       ! e-folding time of liquid drip  tdrip[u,s,l]
     >     tblow,       ! e-folding time of snow blowoff tblow[u,s,l]
     >     wliqmax,     ! max amount of intercepted liquid wliq[u,s,l]max
     >     wsnomax,     ! max amount of intercepted snow   wsno[u,s,l]max
     >     wliqmin,     ! min amount of intercepted liquid (same name for u,s,l)
     >     wsnomin      ! min amount of intercepted snow (same name for u,s,l)
c
      real       
     >  tveg(npoi),     ! temperature of veg component t[u,s,l]
     >  wliq(npoi),     ! intercepted liquid amount wliq[u,s,l] (returned)
     >  wsno(npoi),     ! intercepted snow amount wsno[u,s,l] (returned)
     >  xai(npoi),      ! lai, sai, lai+sai for upper leaves/stems,lower veg
     >  pflux(npoi),    ! ht flux due to adjust of intercep precip (returned)
     >  rain(npoi),     ! rainfall rate. Input: above veg, Output: below veg
     >  train(npoi),    ! temperature of rain. (returned)
     >  snow(npoi),     ! snowfall rate. Input: above veg, output: below veg
     >  tsnow(npoi)     ! temperature of snow (returned)
c
c local variables:
c
      integer i         ! loop indice
c
      real rwork,       ! 1/dtime
     >     x,           ! work variable
     >     rwork2,      ! work variable: ch2o - cice
     >     dw           ! correction: freezing liguid or melting snow
c
      real fint(npoi),  ! precip fraction intercepted by unit leaf/stem area
     >     drip(npoi),  ! rate of liquid drip
     >     blow(npoi)   ! rate of snow blowoff
c
c ---------------------------------------------------------------------
c
c calculate fint, the intercepted precip fraction per unit
c leaf/stem area -- note 0.5 * lai or sai (similar to irrad)
c 
      do 50 i = 1, npoi
c
        if (xai(i).ge.epsilon) then
          fint(i) = ( 1.-exp(-0.5*xai(i)) )/ xai(i)
        else
          fint(i) = 0.5
        endif
c
   50 continue
c
c step intercepted liquid and snow amounts due to drip/blow,
c intercepted rainfall/snowfall, and min/max limits. also 
c adjust temperature of intercepted precip to current veg temp,
c storing the heat needed to do this in pflux for use in turvap
c 
c without these pfluxes, the implicit turvap calcs could not
c account for the heat flux associated with precip adjustments,
c especially changes of phase (see below), and so could not
c handle equilibrium situations such as intercepted snowfall
c being continuously melted by warm atmos fluxes, with the veg 
c temp somewhat lower than the equil atmos temp to supply heat
c that melts the incoming snow; (turvap would just change veg 
c temp to atmos equil, with little sensible heat storage...then
c final phase adjustment would return veg temp to melt point)
c
c the use of the current (ie, previous timestep's) veg temp 
c gives the best estimate of what this timestep's final temp
c will be, at least for steady conditions
c
      rwork = 1. / dtime
c
      do 100 i=1,npoi
c    
c liquid
c
        drip(i) = xai(i)*wliq(i)/tdrip
        wliq(i) = wliq(i) * (1.-dtime/tdrip)
c
        wliq(i) = wliq(i) + dtime*rain(i)*fint(i)
        pflux(i) = rain(i)*fint(i) * (tveg(i)-train(i))*ch2o
        rain(i) = rain(i)*(1.-xai(i)*fint(i))
c
        x = wliq(i)
        wliq(i) = min (wliq(i), wliqmax)
        if (wliq(i).lt.wliqmin) wliq(i) = 0.
        drip(i) = drip(i) + xai(i)*(x-wliq(i))*rwork
c
c snow
c
        blow(i) = xai(i)*wsno(i)/tblow
        wsno(i) = wsno(i) * (1.-dtime/tblow)
c
        wsno(i) = wsno(i) + dtime*snow(i)*fint(i)
        pflux(i) = pflux(i) + snow(i)*fint(i) * (tveg(i)-tsnow(i))*cice
        snow(i) = snow(i)*(1.-xai(i)*fint(i))
c
        x = wsno(i)
        wsno(i) = min (wsno(i), wsnomax)
        if (wsno(i).lt.wsnomin) wsno(i) = 0. 
        blow(i) = blow(i) + xai(i)*(x-wsno(i))*rwork
c
  100 continue
c
c change phase of liquid/snow below/above melt point, and add
c required heat to pflux (see comments above). this will only
c affect the precip intercepted in this timestep, since original
c wliq, wsno must have been ge/le melt point (ensured in later
c call to cascad2/steph2o2)
c
      rwork2 = ch2o - cice
c
      do 300 i=1,npoi
c
c liquid below freezing
c
        dw = 0.
        if (tveg(i).lt.tmelt)  dw = wliq(i)
c
        pflux(i) = pflux(i)
     >           + dw * (rwork2*(tmelt-tveg(i)) - hfus) * rwork
        wliq(i) = wliq(i) - dw
        wsno(i) = wsno(i) + dw
c
c snow above freezing
c
        dw = 0.
        if (tveg(i).gt.tmelt)  dw = wsno(i)
c
        pflux(i) = pflux(i)
     >           + dw * (rwork2*(tveg(i)-tmelt) + hfus) * rwork
        wsno(i) = wsno(i) - dw
        wliq(i) = wliq(i) + dw
c
  300 continue
c
c adjust rainfall, snowfall below veg for interception 
c and drip, blowoff
c
      call mix (rain,train, rain,train, drip,tveg, vzero,vzero)
      call mix (snow,tsnow, snow,tsnow, blow,tveg, vzero,vzero)
c
      return
      end
      subroutine steph2o2 (tveg,wliq,wsno,fveg,xai,rliq,fvapw,cveg)
c ---------------------------------------------------------------------
c
c removes evaporation from intercepted h2o, and does final
c heat-conserving adjustment for any liquid/snow below/above
c melt point, for one veg component
c
      include 'implicit.h'
c
      include 'compar.h'
      include 'comatm.h'
      include 'com1d.h'
c
c Arguments (all arguments are supplied unless otherwise noted)
c
      real cveg        ! specific heat of veg component ch[u,s,l] 
c
      real      
     >  tveg(npoi),    ! temperature of veg component t[u,s,l] (returned)
     >  wliq(npoi),    ! intercepted liquid amount wliq[u,s,l] (returned)
     >  wsno(npoi),    ! intercepted snow amount wsno[u,s,l] (returned)
     >  fveg(npoi),    ! fractional areal coverage, fu or (1-fi)*fl
     >  xai(npoi),     ! lai, sai, lai+sai for upper leaves/stems,lower veg
     >  rliq(npoi),    ! ratio of area wetted by liquid to total wetted area
     >  fvapw(npoi)    ! wetted evap h2o flx per leaf/stem area fvap[uw,s,lw]
c
c local variables
c
      integer i        ! loopi indice
c
      real zm,         ! to compute corrective fluxes
     >     rwork,      ! 1/specific heat of fusion 
     >     chav        ! average specific heat for veg, liw and snow
c
      real dh(npoi),   ! correct heat flux for liquid below melt point and opposite
     >     dw(npoi)    ! correct water flux for liquid below melt point and opposite
c
c
      include 'comsat.h'
c
c ---------------------------------------------------------------------
c
c step intercepted h2o due to evaporation/sublimation.
c (fvapw already has been multiplied by fwet factor in turvap,
c so it is per unit leaf/stem area.)
c
c due to linear fwet factors (see comments in fwetcal) and
c the cap on suw,ssw,slw in turvap, evaporation in one timestep
c should hardly ever make wliq or wsno negative -- but if this
c happens, compensate by increasing vapor flux from atmosphere, 
c and decreasing sensib heat flux from atmos (the former is
c dangerous since it could suck moisture out of a dry atmos,
c and both are unphysical but do fix the budget) tveg in hvapf
c and hsubf should be pre-turvap-timestep values, but are not
c
      do 100 i = 1, npoi
c
        wliq(i) = wliq(i) - dtime *     rliq(i)  * fvapw(i)
        wsno(i) = wsno(i) - dtime * (1.-rliq(i)) * fvapw(i)
c
c check to see if predicted wliq or wsno are less than zero
c
        if ((wliq(i).lt.0. or. wsno(i).lt.0.)
     >      .and. fveg(i)*xai(i).gt.0. )  then
c
c         write (*,9999) i, wliq(i), wsno(i)
c9999     format(' ***warning: wliq<0 or wsno<0 -- steph2o2 9999',
c    >           ' i, wliq, wsno:',i4, 2f12.6)
c
c calculate corrective fluxes
c
          zm = max (-wliq(i), 0.) * fveg(i) * xai(i) / dtime
          fvapa(i) = fvapa(i) + zm
          fsena(i) = fsena(i) - zm*hvapf(tveg(i),ta(i))
          wliq(i) = max (wliq(i), 0.)
c
          zm = max (-wsno(i), 0.) * fveg(i) * xai(i) / dtime
          fvapa(i) = fvapa(i) + zm
          fsena(i) = fsena(i) - zm*hsubf(tveg(i),ta(i))
          wsno(i) = max (wsno(i), 0.)
c
        endif
c
  100 continue
c
c final heat-conserving correction for liquid/snow below/above
c melting point
c
      rwork = 1. / hfus
c
      do 200 i=1,npoi
c
        chav = cveg + ch2o*wliq(i) + cice*wsno(i)
c
c correct for liquid below melt point
c
c (nb: if tveg > tmelt or wliq = 0, nothing changes.)
c
        if (tveg(i).lt.tmelt .and. wliq(i).gt.0.0) then
          dh(i) = chav*(tmelt - tveg(i))
          dw(i) = min (wliq(i), max (0., dh(i)*rwork))
          wliq(i) = wliq(i) - dw(i)
          wsno(i) = wsno(i) + dw(i) 
          chav = cveg + ch2o*wliq(i) + cice*wsno(i)
          tveg(i) = tmelt - (dh(i)-hfus*dw(i))/chav
        endif
c
c correct for snow above melt point
c
c (nb: if tveg < tmelt or wsno = 0, nothing changes.)
c
        if (tveg(i).gt.tmelt .and. wsno(i).gt.0.0) then
          dh(i) = chav*(tveg(i) - tmelt)
          dw(i) = min (wsno(i), max (0., dh(i)*rwork))
          wsno(i) = wsno(i) - dw(i)
          wliq(i) = wliq(i) + dw(i)
          chav = cveg + ch2o*wliq(i) + cice*wsno(i)
          tveg(i) = tmelt + (dh(i)-hfus*dw(i))/chav
        endif
c
  200 continue
c
      return
      end
c
c
c ---------------------------------------------------------------------
      subroutine cascade
c ---------------------------------------------------------------------
c
c steps intercepted h2o due to drip, precip, and min/max limits
c
c calls steph2o for upper leaves, upper stems and lower veg in
c iurn, adjusting precips at each level
c
      include 'implicit.h'
c
      include 'compar.h'
      include 'comatm.h'
      include 'comveg.h'
      include 'com1d.h'
c
c local variables
c
      integer i            ! loop indice
c
      real twet3           ! Function: wet bulb temperature (K)
      real twetbulb        ! wet bulb temperature (K)
c    
      real
     >  xai(npoi),         !lai and/or sai for veg component
                           ! (allows steph2o to work on any veg component)
     >  rain(npoi),        !rainfall at appropriate level (modified by steph2o)
     >  train(npoi),       !temperature of rain (modified by steph2o)  
     >  snow(npoi),        !snowfall at appropriate level (modified by steph2o)
     >  tsnow(npoi),       !temperature of snow (modified by steph2o)
     >  x1(npoi),        ! 
     >  x2(npoi),        ! 
     >  x3(npoi),        ! 
     >  x4(npoi)         ! 
c
c adjust rainfall and snowfall rates at above-tree level
c
c set wliqmin, wsnomin -- unlike wliq*max, wsno*max, these are
c part of the lsx numerical method and not from the vegetation
c database, and they are the same for all veg components
c
c the value 0.0010 should be small compared to typical precip rates
c times dtime to allow any intercepted h2o to be initiated, but
c not too small to allow evap rates to reduce wliq*, wsno* to
c that value in a reasonable number of time steps
c
      wliqmin = 0.0010 * (dtime/3600.) * (wliqumax / 0.2)
      wsnomin = 0.0010 * (dtime/3600.) * (wsnoumax / 2.0)
c
      do 50 i=1,npoi
        rainu(i) = raina(i)
c
c set rain temperature to the wet bulb temperature
c
        if (ta(i) .gt. tmelt) then
           twetbulb = twet3( ta(i), qa(i), psurf(i) )
        else
           twetbulb = tmelt
        endif
        trainu(i) = max (twetbulb, tmelt)
        x1(i) = 0.0
        x2(i) = max (t12(i), tmelt)
   50 continue
c
      call mix (rainu,trainu, rainu,trainu, x1,x2, vzero,vzero)
c
      do 52 i=1,npoi
        snowu(i) = snowa(i)
        tsnowu(i) = min (ta(i), tmelt)
        x1(i) = 0.0
        x2(i) = min (t12(i), tmelt)
   52 continue
c
      call mix (snowu,tsnowu, snowu,tsnowu, x1,x2, vzero,vzero)
c
c set up for upper leaves
c
      do 100 i = 1, npoi
        xai(i)   = 2.0 * lai(i,2)
        rain(i)  = rainu(i)
        train(i) = trainu(i)
        snow(i)  = snowu(i)
        tsnow(i) = tsnowu(i)
  100 continue
c
c step upper leaves
c
      call steph2o
     >  (tu,  wliqu,  wsnou,  xai,  pfluxu,  rain, train, snow, tsnow,
     >   tdripu, tblowu, wliqumax, wsnoumax, wliqmin, wsnomin)
c
c set up for upper stems
c the upper stems get precip as modified by the upper leaves
c
      do 200 i=1,npoi
        xai(i) = 2.0 * sai(i,2)
  200 continue
c
c step upper stems
c
      call steph2o
     >  (ts,  wliqs,  wsnos,  xai,  pfluxs,  rain, train, snow, tsnow,
     >   tdrips, tblows, wliqsmax, wsnosmax, wliqmin, wsnomin)
c
c adjust rainfall and snowfall rates at below-tree level
c allowing for upper-veg interception/drip/belowoff
c
      do 300 i=1,npoi
        x1(i) = fu(i)*rain(i)
        x2(i) = (1.-fu(i))*rainu(i)
        x3(i) = 0.0
        x4(i) = max (t34(i), tmelt)
  300 continue
c
      call mix (rainl,trainl, x1,train, x2,trainu, x3,x4)
c
      do 310 i=1,npoi
        x1(i) = fu(i)*snow(i)
        x2(i) = (1.-fu(i))*snowu(i)
        x3(i) = 0.0
        x4(i) = min (t34(i), tmelt)
  310 continue
c
      call mix (snowl,tsnowl, x1,tsnow, x2,tsnowu, x3,x4)
c
c set up for lower veg
c
      do 400 i = 1, npoi
        xai(i)   = 2.0 * (lai(i,1) + sai(i,1))
        rain(i)  = rainl(i)
        train(i) = trainl(i)
        snow(i)  = snowl(i)
        tsnow(i) = tsnowl(i)
  400 continue
c
c step lower veg
c
      call steph2o
     >  (tl,  wliql,  wsnol,  xai,  pfluxl,  rain, train, snow, tsnow,
     >   tdripl, tblowl, wliqlmax, wsnolmax, wliqmin, wsnomin)
c
c adjust rainfall and  snowfall rates at soil level,
c allowing for lower-veg interception/drip/blowoff
c
      do 500 i=1,npoi
        x1(i) = fl(i) * rain(i)
        x2(i) = (1.-fl(i)) * rainl(i)
  500 continue
c
      call mix (raing,traing, x1,train, x2,trainl, vzero,vzero)
c
      do 510 i=1,npoi
        x1(i) = fl(i) * snow(i)
        x2(i) = (1.-fl(i)) * snowl(i)
  510 continue
c
      call mix (snowg,tsnowg, x1,tsnow, x2,tsnowl, vzero,vzero)
c
      return
      end
c
c ---------------------------------------------------------------------
      subroutine cascad2
c ---------------------------------------------------------------------
c
c at end of timestep, removes evaporation from intercepted h2o,
c and does final heat-conserving adjustment for any liquid/snow 
c below/above melt point. calls steph2o2 for upper leaves, 
c upper stems and lower veg in turn.
c
      include 'implicit.h'
c
      include 'compar.h'
      include 'comsno.h'
      include 'comveg.h'
      include 'com1d.h'
c
c local variables
c
      integer i           ! loop indice
c
      real fveg(npoi),    ! fractional areal coverage of veg component
     >     xai(npoi)      ! lai and/or sai for veg component
c
c ---------------------------------------------------------------------
c
c set up for upper leaves
c
      do 100 i=1,npoi
        fveg(i) = fu(i)
        xai(i) = 2.0 * lai(i,2)
  100 continue
c
c step upper leaves
c
      call steph2o2 (tu,wliqu,wsnou,fveg,xai,rliqu,fvapuw,chu)
c
c set up for upper stems
c
      do 200 i=1,npoi
        fveg(i) = fu(i)
        xai(i) = 2.0 * sai(i,2)
  200 continue
c
c step upper stems
c
      call steph2o2 (ts,wliqs,wsnos,fveg,xai,rliqs,fvaps,chs)
c
c set up for lower veg
c
      do 400 i=1,npoi
        fveg(i) = (1.-fi(i))*fl(i)
        xai(i) = 2.0 * (lai(i,1) + sai(i,1))
  400 continue
c
c step lower veg
c
      call steph2o2 (tl,wliql,wsnol,fveg,xai,rliql,fvaplw,chl)
c
      return
      end
