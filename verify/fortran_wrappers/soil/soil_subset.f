      subroutine setsoi
c ---------------------------------------------------------------------
c
c sets diagnostic soil quantities
c
      include 'implicit.h'
c
      include 'compar.h'
      include 'comatm.h'
      include 'comsno.h'
      include 'comsoi.h'
      include 'comveg.h'
c
c Local variables
c
      integer i, k,           ! loop indices
     >        msand,          ! % of sand in grid point
     >        mclay           ! % of clay in grid point
c
      real fsand,             ! fraction of sand in grid point
     >     fsilt,             ! fraction of silt in grid point
     >     fclay,             ! fraction of clay in grid point
* MEM: added forganic for organic soils.
     >     forganic,          ! fraction of organic soil in grid point
     >     powliq,            ! liquid water content in fraction of soil depth
     >     powice,            ! ice water content in fraction of soil depth
     >     zcondry,           ! dry-soil conductivity
     >     zvap,              ! latent heat of vaporisation at soil temp
     >     zsub,              ! latent heat of sublimation at soil temp
     >     zwpud,             ! fraction of soil surface covered by puddle
*     >     zwpmax,            ! assumed maximum value of zwpud
     >     zwsoi,             ! volumetric water content of top soil layer 
     >     rwork1,
     >     rwork2
c
      include 'comsat.h'    
c
c set soil layer quantities
c
      do 100 k = 1, nsoilay
c
        do 110 i = 1, npoi
c
c Convert input sand and clay percents to fractions
c
          msand = nint(sand(i,k))
          mclay = nint(clay(i,k))
c
          fsand = 0.01 * msand
          fclay = 0.01 * mclay
          fsilt = 0.01 * (100 - msand - mclay)
c
c update thermal conductivity (w m-1 k-1)
c
c based on c = c1**v1 * c2**v2 * c3**v3 * c4**v4 where c1,c2..
c are conductivities of soil grains, air, liquid and ice
c respectively, and v1,v2... are their volume fractions 
c (so v1 = 1-p where p is the porosity, and v1+v2+v3+v4 = 1).
c then condry = c1**(1-p) * c2**p  is the dry-soil
c conductivity, and c = condry * (c3/c2)**v3 * (c4/c2)**v4, 
c where c2 = conductivity of air = .025 w m-1 k-1.
c however this formula agrees better with williams+smith
c table 4 for wet (unfrozen) sand and clay if c2 is decreased
c to ~.005. (for peat in next section, ok if c2 = .025).
c also see lachenbruch etal,1982,jgr,87,9301 and refs therein.
c
          powliq = poros(i,k) * wsoi(i,k) * (1. - wisoi(i,k))
          powice = poros(i,k) * wisoi(i,k)
c
          zcondry = fsand * 0.300 +
     >              fsilt * 0.265 +
     >              fclay * 0.250 ! +
* M. El Maayar added this to account for contribution of organic soils
c     >              forganic * 0.026
c
          consoi(i,k) = zcondry * ((0.56*100.)**powliq)
     >                          * ((2.24*100.)**powice)
c
 110    continue
c
 100  continue
c
c set qglif - the fraction of soil sfc evaporation from soil liquid,
c soil ice, puddle liquid, and puddle ice (relative to total sfc evap)
c
c zwpud:   fraction of surface area covered by puddle (range: 0 - zwpmax)
c zwpmax:  maximum value of zwpud (currently assumed to be 0.5)
c 1-zwpud: fraction of surface area covered by soil (range: (1-zwpmax) - 1.0)
c zwsoi:   volumetric water content of top soil layer (range: 0 - 1.0)
c
c qglif(i,1): fraction of soil evap (fvapg) from soil liquid
c qglif(i,2): fraction of soil evap (fvapg) from soil ice
c qglif(i,3): fraction of soil evap (fvapg) from puddle liquid
c qglif(i,4): fraction of soil evap (fvapg) from puddle ice
c
      do 200 i = 1, npoi
c
*        zwpmax = 0.5
        zwpud = max (0.0, min (zwpmax, zwpmax*(wpud(i)+wipud(i))/wpudmax) )
        zwsoi = (1. - wisoi(i,1)) * wsoi(i,1) + wisoi(i,1)
c
        if (zwsoi.ge.epsilon) then
c
          rwork1 = 1./zwsoi
c
          if (zwpud.ge.epsilon) then
            rwork2 = 1./(wpud(i) + wipud(i))
            qglif(i,1) = (1. - zwpud) * (1. - wisoi(i,1)) * wsoi(i,1) * rwork1
            qglif(i,2) = (1. - zwpud) * wisoi(i,1) * rwork1
            qglif(i,3) = zwpud * wpud(i) * rwork2
            qglif(i,4) = zwpud * wipud(i) * rwork2
          else
            qglif(i,1) = (1. - wisoi(i,1)) * wsoi(i,1) * rwork1
            qglif(i,2) = wisoi(i,1) * rwork1
            qglif(i,3) = 0.0
            qglif(i,4) = 0.0
          endif
c
        else
c
c for a 100% dry soil surface, assign all soil evap to the puddles.
c Note that for small puddle sizes, this could lead to negative
c puddle depths. However, for a 100% dry soil with small puddles,
c evaporation is likely to be very small or less than zero
c (condensation), so negative puddle depths are not likely to occur.
c
          if (zwpud.ge.epsilon) then
            rwork2 = 1./(wpud(i) + wipud(i))
            qglif(i,1) = 0.0
            qglif(i,2) = 0.0
            qglif(i,3) = zwpud * wpud(i) * rwork2
            qglif(i,4) = zwpud * wipud(i) * rwork2
          else
            if (tsoi(i,1).ge.tmelt) then
c
c above freezing
c
              qglif(i,1) = 0.
              qglif(i,2) = 0.
              qglif(i,3) = 1.
              qglif(i,4) = 0.
c
            else
c
c below freezing
c
              qglif(i,1) = 0.
              qglif(i,2) = 0.
              qglif(i,3) = 0.
              qglif(i,4) = 1.
            endif
          endif
c
        endif
c
c set latent heat values
c
        zvap = hvapf (tsoi(i,1), ta(i))
        zsub = hsubf (tsoi(i,1), ta(i))
c
        hvasug(i) = (qglif(i,1) + qglif(i,3)) * zvap +
     >              (qglif(i,2) + qglif(i,4)) * zsub 
c
        hvasui(i) = hsubf(tsno(i,1),ta(i))
c
 200  continue   
c
      return
      end
      subroutine soilh2o (owsoi, fwtop, fwpud, fsqueez, wflo)
c ---------------------------------------------------------------------
c
c sets up call to tridia to solve implicit soil moisture eqn,
c using soil temperatures in wsoi (in comsoi)
c
c lower bc can be no h2o flow or free drainage, set by bperm below
c
      include 'implicit.h'
c
      include 'compar.h'
      include 'comsoi.h'
      include 'com1d.h'
c
c Arguments : all are supplied except wflo (returned):
c
      real owsoi(npoi,nsoilay),  ! soil moistures at start of timestep
     >     fwtop(npoi),          ! evaporation from top soil layer
     >     fwpud(npoi),          ! h2o flux into top layer (infiltration from puddle)
     >     fsqueez(npoi),        ! excess water at end of time step in soil column  
     >     wflo(npoi,nsoilay+1)  ! downward h2o flow across layer boundaries
c
c local variables
c     
      integer k, i,    ! loop indices
     >        km1,
     >        kka, 
     >        kkb 
c
      integer 
     >  m(npoi),                ! exponents 
     >  n(npoi)
c
      real dmin,                ! minimum diffusivity for dry soils (m**2 s-1) 
     >     rimp,                ! implicit fraction of the calculation (0 to 1)
*     >     bperm,               ! = 0 for impermeable (no drainage) lower bc,
     >                          ! = 1 for free drainage lower bc
     >     zbex, z, dt, zz
c 
      real hsoim(nsoilay+1)  ! vertical distances between centers of layers
c
      real
     >  wsoim(npoi,nsoilay+1),    ! interpolated moisture values at layer boundaries
     >  wsoia(npoi,nsoilay+1),    ! 
     >  wsoib(npoi,nsoilay+1),    ! '
     >  weim(npoi,nsoilay+1),     ! '
     >  weip(npoi,nsoilay+1),     ! '
     >  a(npoi),                  ! intermediate terms (const for each pt)
     >  b(npoi),                  ! 
     >  bwn(npoi),                ! 
     >  bwn1(npoi),               ! 
     >  e(npoi,nsoilay+1),        ! intermediate terms in algebraic devel 
     >  f(npoi,nsoilay+1),        ! '
     >  g(npoi,nsoilay+1),        ! '
     >  d1(npoi,nsoilay),         ! diagonals of tridiagonal systems of equations 
     >  d2(npoi,nsoilay),         !  '
     >  d3(npoi,nsoilay),         !  '
     >  rhs(npoi,nsoilay),        ! right-hand sides of systems of equations
     >  w1(npoi,nsoilay),         ! work arrays needed by tridia
     >  w2(npoi,nsoilay)          !  '
c
      save dmin, rimp
      data dmin, rimp /1.e-9, 1.0/
c
c set lower boundary condition for the soil
c (permeability of the base)
c
c     bperm = 0.00  ! e.g. fully impermeable base
c     bperm = 1.00  ! e.g. fully permeable base
c
*      bperm = 0.10
c
c set level vertical distances, interpolated moistures, and
c interpolation weights
c
c top layer
c
      k = 1
c
      do 100 i = 1, npoi
c
        hsoim(k) = 0.5 * hsoi(k)
c
        weip(i,k) = 0.0
        weim(i,k) = 1.0
c
        wsoim(i,k) = wsoi(i,k)
        wsoia(i,k) = min (wsoim(i,k), 1.0)
        wsoib(i,k) = min (wsoim(i,k), 1.0)
c
  100 continue
c
c middle layers
c
      do 110 k = 2, nsoilay
c
        do 120 i = 1, npoi
c
          hsoim(k) = 0.5 * (hsoi(k-1) + hsoi(k))
c
          weip(i,k) = 0.5 * hsoi(k) / hsoim(k) 
          weim(i,k) = 1.0 - weip(i,k)
c
          wsoim(i,k) = weim(i,k) * wsoi(i,k-1) + weip(i,k) * wsoi(i,k)
          wsoia(i,k) = min (wsoim(i,k), 1.0)
          wsoib(i,k) = min (wsoim(i,k), 1.0)
c
  120   continue
c
  110 continue
c
c bottom layer
c
      k = nsoilay + 1
c
      do 130 i = 1, npoi
c
        hsoim(k) = 0.5 * hsoi(k-1)
c
        weip(i,k) = 1.0
        weim(i,k) = 0.0
c
        wsoim(i,k) = wsoi(i,k-1)
        wsoia(i,k) = min (wsoim(i,k), 1.0)
        wsoib(i,k) = min (wsoim(i,k), 1.0)
c
  130 continue
c
c set intermediate quantities e,f,g. these are terms in the
c expressions for the fluxes at boundaries between layers,
c so are zero for k=1. use bwn1 to account for minimum 
c diffusivity dmin. bperm is used for k=nsoilay+1 to set the
c type of the lower bc.
c
c top layer
c
      k = 1
c
      call const (e(1,k), npoi, 0.0)
      call const (f(1,k), npoi, 0.0)
      call const (g(1,k), npoi, 0.0)
c
c middle layers
c
      do 200 k = 2, nsoilay
c
        do 210 i = 1, npoi
c
c now that hydraul, suction and ibex can vary with depth,
c use averages of surrounding mid-layer values
c
c (see notes 8/27/93)
c
          a(i) = weim(i,k) * hydraul(i,k-1) +
     >           weip(i,k) * hydraul(i,k  )
c
          b(i) = weim(i,k) * hydraul(i,k-1) * 
     >           suction(i,k-1) * bex(i,k-1) +
     >           weip(i,k) * hydraul(i,k  ) *
     >           suction(i,k  ) * bex(i,k  )
c
          zbex = weim(i,k) * bex(i,k-1) +
     >           weip(i,k) * bex(i,k  ) 
c
          m(i) = 2 * nint(zbex) + 3
          n(i) =     nint(zbex) + 2
c
          bwn1(i) = b(i) * (wsoib(i,k)**(n(i)-1))
          bwn(i)  = bwn1(i) * wsoib(i,k)
c
          if (bwn(i).lt.dmin) bwn1(i) = 0.0
          bwn(i) = max (bwn(i), dmin)
c
          e(i,k) =  (-1.+rimp*m(i))*a(i)*(wsoia(i,k)**m(i))
     >            + ((1.-rimp)*bwn(i) - rimp*n(i)*bwn1(i)*wsoib(i,k))
     >              * (wsoi(i,k)-wsoi(i,k-1)) / hsoim(k)
c
          f(i,k) = - rimp*m(i)*a(i)*(wsoia(i,k)**(m(i)-1))
     >             + rimp*n(i)*bwn1(i)
     >               * (wsoi(i,k)-wsoi(i,k-1)) / hsoim(k)
c
          g(i,k) = rimp*bwn(i)
c
  210     continue
c
  200 continue
c
c bottom layer
c
      k = nsoilay + 1
c
      do 220 i = 1, npoi
c
        a(i) = hydraul(i,nsoilay) 
        b(i) = hydraul(i,nsoilay)*suction(i,nsoilay)*ibex(i,nsoilay)
c
        m(i) = 2*ibex(i,nsoilay) + 3
        n(i) = ibex(i,nsoilay)   + 2
c
        e(i,k) =                -a(i)*(wsoia(i,k)**m(i))*bperm
        f(i,k) = 0.0
        g(i,k) = 0.0
c
  220 continue
c
c deduce all e,f,g in proportion to the minimum of the two 
c adjacent layers' (1-wisoi), to account for restriction of flow
c by soil ice. this will cancel in loop 300  with the factor 
c 1-wisoi in (one of) the layer's porosflo, even if wisoi=1 by 
c the use of epsilon limit. so a layer with wisoi=1 will form a 
c barrier to flow of liquid, but still have a predicted wsoi
c
      do 230 k = 1, nsoilay+1
c
        kka = max (k-1,1)
        kkb = min (k,nsoilay)
c
        do 240 i=1,npoi
c
c multiply by an additional factor of 1-wisoi for stability
c
          z = max(0.,1.-max(wisoi(i,kka),wisoi(i,kkb)))**2
c
          e(i,k) = z * e(i,k)
          f(i,k) = z * f(i,k)
          g(i,k) = z * g(i,k)
c
  240   continue
c
  230 continue
c
c set matrix diagonals and right-hand sides
c
      do 300 k = 1, nsoilay
c
        do 310 i = 1, npoi
c
          dt = dtime / (porosflo(i,k)*hsoi(k))
          d1(i,k) = dt*(   f(i,k)*0.5*hsoi(k)/hsoim(k)
     >                   - g(i,k)/hsoim(k) )
          rhs(i,k) = wsoi(i,k) + dt*( e(i,k+1) - e(i,k) )
c
  310   continue
c
        if (k.eq.1) then
c
          do 320 i=1,npoi
c
            dt = dtime / (porosflo(i,k)*hsoi(k))
            rhs(i,k) = rhs(i,k) + dt*(fwtop(i)+fwpud(i))/rhow
c
  320     continue
c
        endif
c
        if (k.lt.nsoilay) then
c
          km1 = max (k-1,1)
c
          do 330 i=1,npoi
c
            dt = dtime / (porosflo(i,k)*hsoi(k))
            d2(i,k) = 1. + dt*( - f(i,k+1)*0.5*hsoi(k+1)/hsoim(k+1)
     >                          + f(i,k)  *0.5*hsoi(km1)/hsoim(k)
     >                          + g(i,k+1)/hsoim(k+1)
     >                          + g(i,k)  /hsoim(k) )
            d3(i,k) = dt*( - f(i,k+1)*0.5*hsoi(k)/hsoim(k+1)
     >                     - g(i,k+1)            /hsoim(k+1) )
c
  330     continue
c
        else if (k.eq.nsoilay) then
c
          do 340 i=1,npoi
c
            dt = dtime / (porosflo(i,k)*hsoi(k))
            d2(i,k) = 1. + dt*( - f(i,k+1)
     >                          + f(i,k)  *0.5*hsoi(k-1)/hsoim(k)
     >                          + g(i,k)  /hsoim(k) )
            d3(i,k) = 0.0
c
  340     continue
c
        endif
c
  300 continue
c
c solve the systems of equations
c
      call tridia (npoi, npoi, nsoilay, d1,d2,d3, rhs, wsoi, w1,w2)
c
      do 400 i = 1, npoi
c
        fsqueez(i) = 0.0
        wflo(i,nsoilay+1) = - rhow * e(i,nsoilay+1)
c
  400 continue
c
      do 500 k = nsoilay, 1, -1
c
        do 510 i = 1, npoi
c
          zz = rhow * poros(i,k) * 
     >         max(epsilon, (1.-wisoi(i,k))) * hsoi(k)
c
          wsoi(i,k) = wsoi(i,k) + dtime * fsqueez(i) / zz 
          fsqueez(i) = max (wsoi(i,k)-1.,0.) * zz / dtime
          wsoi(i,k) = min (wsoi(i,k),1.)
c
          wflo(i,k) = wflo(i,k+1) + (wsoi(i,k)-owsoi(i,k)) * zz / dtime
c
  510   continue
c
  500 continue
c
c step puddle liquid due to fsqueez and fwpud
c
c also subtract net puddle-to-top-layer flux from wflo(i,1),
c since puddle and top soil layer are lumped together in soilheat
c so upper wflo should be external flux only (evap/condens)

      do 600 i = 1, npoi
c
        wpud(i)   = wpud(i)   + (fsqueez(i) - fwpud(i)) * dtime
        wflo(i,1) = wflo(i,1) + (fsqueez(i) - fwpud(i))
c
  600 continue
c
      return
      end
c
c
c ---------------------------------------------------------------------
      subroutine const (arr, nar, value)
c ---------------------------------------------------------------------
c
c sets all elements of real vector arr to value
c
      include 'implicit.h'
c
c Arguments
c
      integer nar
c     
      real value
      real arr(nar)
c
c Local variables
c
      integer j
c
      do 100 j = 1, nar
        arr(j) = value
 100  continue
c
      return
      end
c
      subroutine tridia (ns, nd, ne, a, b, c, y, x, alpha, gamma)
c ---------------------------------------------------------------------
c
      include 'implicit.h'
c
      include 'compar.h'
c
c     purpose:
c     to compute the solution of many tridiagonal linear systems.
c
c      arguments:
c
c      ns ..... the number of systems to be solved.
c
c      nd ..... first dimension of arrays (ge ns).
c
c      ne ..... the number of unknowns in each system.
c               this must be > 2. second dimension of arrays.
c
c      a ...... the subdiagonals of the matrices are stored
c               in locations a(j,2) through a(j,ne).
c
c      b ...... the main diagonals of the matrices are stored
c               in locations b(j,1) through b(j,ne).
c
c      c ...... the super-diagonals of the matrices are stored in
c               locations c(j,1) through c(j,ne-1).
c
c      y ...... the right hand side of the equations is stored in
c               y(j,1) through y(j,ne).
c
c      x ...... the solutions of the systems are returned in
c               locations x(j,1) through x(j,ne).
c
c      alpha .. work array dimensioned alpha(nd,ne)
c
c      gamma .. work array dimensioned gamma(nd,ne)
c
c       history:  based on a streamlined version of the old ncar
c                 ulib subr trdi used in the phoenix climate
c                 model of schneider and thompson (j.g.r., 1981).
c                 revised by starley thompson to solve multiple
c                 systems and vectorize well on the cray-1.
c                 later revised to include a parameter statement
c                 to define loop limits and thus enable cray short
c                 vector loops.
c
c       algorithm:  lu decomposition followed by solution.
c                   note: this subr executes satisfactorily
c                   if the input matrix is diagonally dominant
c                   and non-singular.  the diagonal elements are
c                   used to pivot, and no tests are made to determine
c                   singularity. if a singular or numerically singular
c                   matrix is used as input a divide by zero or
c                   floating point overflow will result.
c
c       last revision date:      4 february 1988
c
c
c Arguments
c
      integer ns,     ! number of systems to be solved.
     >        nd,     ! first dimension of arrays (ge ns)
     >        ne      ! number of unknowns in each system. (>2)
      
      real 
     >  a(nd,ne),     ! subdiagonals of matrices stored in a(j,2)...a(j,ne).
     >  b(nd,ne),     ! main diagonals of matrices stored in b(j,1)...b(j,ne).
     >  c(nd,ne),     ! super-diagonals of matrices stored in c(j,1)...c(j,ne-1).
     >  y(nd,ne),     ! right hand side of equations stored in y(j,1)...y(j,ne).
     >  x(nd,ne),     ! solutions of the systems returned in x(j,1)...x(j,ne).
     >  alpha(nd,ne), ! work array 
     >  gamma(nd,ne)  ! work array
c
c local variables
c
      integer nm1,    !
     >  j, i, ib      ! loop indices

c
      nm1 = ne-1
c
c obtain the lu decompositions
c
      do 10 j=1,ns
         alpha(j,1) = 1./b(j,1)
         gamma(j,1) = c(j,1)*alpha(j,1)
   10 continue
      do 11 i=2,nm1
         do 12 j=1,ns
            alpha(j,i) = 1./(b(j,i)-a(j,i)*gamma(j,i-1))
            gamma(j,i) = c(j,i)*alpha(j,i)
   12    continue
   11 continue
c
c solve
c
      do 20 j=1,ns
         x(j,1) = y(j,1)*alpha(j,1)
   20 continue
      do 21 i=2,nm1
         do 22 j=1,ns
            x(j,i) = (y(j,i)-a(j,i)*x(j,i-1))*alpha(j,i)
   22    continue
   21 continue
      do 23 j=1,ns
         x(j,ne) = (y(j,ne)-a(j,ne)*x(j,nm1))/
     >             (b(j,ne)-a(j,ne)*gamma(j,nm1))
   23 continue
      do 24 i=1,nm1
         ib = ne-i
         do 25 j=1,ns
            x(j,ib) = x(j,ib)-gamma(j,ib)*x(j,ib+1)
   25    continue
   24 continue
c
      return
      end
      subroutine soilheat (otsoi, owsoi, c0pud, fhtop, wflo, c1pud)
c ---------------------------------------------------------------------
c
c        sets up call to tridia to solve implicit soil/ice heat 
c        conduction, using layer temperatures in tsoi (in comsoi).
c        the heat flux due to liquid flow previously calculated
c        in soilh2o is accounted for. lower bc is conductive flux = 0
c        for soil (although the flux due to liquid drainage flow can
c        be > 0)
c
      include 'implicit.h'
c
      include 'compar.h'
      include 'comsoi.h'
c
c Arguments
c
      real
     >  otsoi(npoi,nsoilay),    ! soil/ice temperatures at start of timestep (redundant
c                                 with tsoi, but passed to be consistent with soilh2o)
     >  owsoi(npoi,nsoilay),    ! soil moistures at start of timestep (before soilh2o)
     >  c0pud(npoi,nsoilay),    ! layer heat capacity due to puddles (=0 except for top)
     >  c1pud(npoi,nsoilay),    ! updated c0pud
     >  fhtop(npoi),            ! heat flux into top layer from atmos
     >  wflo(npoi,nsoilay+1)    ! downward h2o  flow across layer boundaries
c
c local variables
c
      integer k, i, km1, kp1    ! loop indices
c
      real rimp,                ! implicit fraction of the calculation (0 to 1)
     >     rwork, rwork1,       ! work variables
     >     rwork2, rwork3,      ! work variables
     >     t
c
      real
     >  whflo(npoi,nsoilay+1),   ! downward heat fluxes across layer bdries due to h2o
     >                           ! movement calculated in soilh2o
     >  con(npoi,nsoilay+1),     ! conduction coeffs between layers
     >  c0(npoi,nsoilay),        ! specific heats at start of timestep
     >  c1(npoi,nsoilay),        ! specific heats at end of timestep
     >  d1(npoi,nsoilay),        ! diagonals of tridiagonal systems of equations
     >  d2(npoi,nsoilay),        !  ''
     >  d3(npoi,nsoilay),        !  ''
     >  rhs(npoi,nsoilay),       ! right-hand sides of systems of equations
     >  w1(npoi,nsoilay),        ! work arrays needed by tridia
     >  w2(npoi,nsoilay)
c
      data rimp /1.0/
c
c set conduction coefficient between layers, and heat fluxes
c due to liquid transport
c
c top layer
c
      k = 1
c
      do 100 i = 1, npoi
        con(i,k) = 0.0
        whflo(i,k) = wflo(i,k) * ch2o * tsoi(i,k)
  100 continue
c
c middle layers
c
      do 110 k = 2, nsoilay
c
        do 120 i = 1, npoi
c
          con(i,k) =  1. / (0.5 * (hsoi(k-1) / consoi(i,k-1) +
     >                             hsoi(k)   / consoi(i,k)))
c
          t = (hsoi(k) * tsoi(i,k-1) + hsoi(k-1) * tsoi(i,k)) /
     >        (hsoi(k-1)             + hsoi(k))
c
          whflo(i,k) = wflo(i,k) * ch2o * t
c
  120   continue
c
  110 continue
c
c bottom layer
c
      k = nsoilay + 1
c
      do 130 i = 1, npoi
        con(i,k) = 0.0
        whflo(i,k) = wflo(i,k) * ch2o * tsoi(i,k-1)
  130 continue
c
c set diagonals of matrix and right-hand side. use old and
c new heat capacities c0, c1 consistently with moisture fluxes
c whflo computed above, to conserve heat associated with 
c changing h2o amounts in each layer
c
      do 200 k = 1, nsoilay
c
        km1 = max (k-1,1)
        kp1 = min (k+1,nsoilay)
c
        do 210 i=1,npoi
c
          rwork1 = (1.-poros(i,k))*csoi(i,k)*rhosoi(i,k)
          rwork2 = poros(i,k)*(1.-wisoi(i,k))*ch2o*rhow
          rwork3 = poros(i,k)*wisoi(i,k)*cice*rhow
c
          c0(i,k) = c0pud(i,k) + 
     >              (   rwork1
     >                + rwork2 * owsoi(i,k)
     >                + rwork3
     >              ) * hsoi(k)
c
          c1(i,k) = c1pud(i,k) +  
     >              (   rwork1
     >                + rwork2 * wsoi(i,k)
     >                + rwork3
     >              ) * hsoi(k)
c
          rwork = dtime/c1(i,k)
c
          d1(i,k) =    - rwork * rimp * con(i,k)
          d2(i,k) = 1. + rwork * rimp * (con(i,k)+con(i,k+1))
          d3(i,k) =    - rwork * rimp * con(i,k+1)
c
          rhs(i,k) = (c0(i,k)/c1(i,k))*tsoi(i,k) + rwork
     >               * ( (1.-rimp)*con(i,k)  *(tsoi(i,km1)-tsoi(i,k))
     >                 + (1.-rimp)*con(i,k+1)*(tsoi(i,kp1)-tsoi(i,k))
     >                 + whflo(i,k) - whflo(i,k+1) )
c
  210   continue
c
        if (k.eq.1) then
          do 220 i=1,npoi
            rhs(i,k) = rhs(i,k) + (dtime/c1(i,k))*fhtop(i)
  220     continue
        endif
c
  200 continue
c
c solve systems of equations
c
      call tridia (npoi, npoi, nsoilay, d1,d2,d3, rhs, tsoi, w1,w2)
c
c deduce downward heat fluxes between layers
c
      call scopy (npoi, fhtop, hflo(1,1))
c
      do 300 k=1,nsoilay
        do 310 i=1,npoi
          hflo(i,k+1) = hflo(i,k) -
     >                  (c1(i,k)*tsoi(i,k) - c0(i,k)*otsoi(i,k)) / dtime
  310   continue
  300 continue
c
      return
      end
      subroutine scopy (nt, arr, brr)
c ---------------------------------------------------------------------
c
c copies array arr to brr,for 1st nt words of arr
c
      include 'implicit.h'
c
c Arguments
c
      integer nt     
      real arr(nt),    ! input
     >     brr(nt)     ! output
c
c Local variables
c
      integer ia
c
      do 100 ia = 1, nt
        brr(ia) = arr(ia)
 100  continue
c
      return
      end
      subroutine wadjust
c ---------------------------------------------------------------------
c
c set wsoi, wisoi to exactly 0 if differ by negligible amount, 
c to protect epsilon logic in soilctl and soilh2o
c
c ice-liquid transformations in soilctl loop 400 can produce very
c small -ve amounts due to roundoff error, and very small -ve or +ve
c amounts can cause (harmless) "underflow" fpes in soilh2o
c
      include 'implicit.h'
c
      include 'compar.h'
      include 'comhyd.h'
      include 'comsoi.h'
      include 'com1d.h'
c 
c local variables
c
      integer k, i
      real ztot0, ztot1
c
      do 100 k = 1, nsoilay
        do 110 i = 1, npoi
c
c initial total soil water
c
         ztot0 = hsoi(k) * poros(i,k) * rhow *
     >           ((1. - wisoi(i,k)) * wsoi(i,k) + wisoi(i,k))
c
c set bounds on wsoi and wisoi
c
         if (wsoi(i,k).lt.epsilon)  wsoi(i,k)  = 0.0
         if (wisoi(i,k).lt.epsilon) wisoi(i,k) = 0.0
c
         wsoi(i,k)  = min (1., wsoi(i,k))
         wisoi(i,k) = min (1., wisoi(i,k))
c
         if (wisoi(i,k).ge.1-epsilon) wsoi(i,k) = 0.0
c
c for diagnosis of total adjustment
c
         ztot1 = hsoi(k) * poros(i,k) * rhow *
     >           ((1. - wisoi(i,k)) * wsoi(i,k) + wisoi(i,k))
c
         gadjust(i) = gadjust(i) + (ztot1 - ztot0) / dtime
c
  110   continue
  100 continue
c
      return
      end


c ---------------------------------------------------------------------
c soil_freeze_thaw_block
c
c Heat-conserving liquid<->ice phase change for each soil layer.
c Carved out from soilctl (soil.f lines 411-472).
c Reads/writes wsoi, wisoi, tsoi (via common /comsoi5/).
c Uses poros, csoi, rhosoi, c1pud, hsoi, porosflo and the scalars
c tmelt, hfus, ch2o, cice, rhow, epsilon.
c Also writes porosflo at the end (just for downstream consistency).
c ---------------------------------------------------------------------
      subroutine soil_freeze_thaw_block(c1pud_arr)
      include 'implicit.h'
      include 'compar.h'
      include 'comsoi.h'

      real c1pud_arr(npoi, nsoilay)

      integer i, k
      real chav, rwork, zwsoi
      real zporos(npoi), dh(npoi), dw(npoi)

      do 400 k = 1, nsoilay
        do 410 i = 1, npoi

          zporos(i) = max(poros(i,k), epsilon)
          rwork = c1pud_arr(i,k)/rhow/hsoi(k)
     >          + (1.-zporos(i))*csoi(i,k)*rhosoi(i,k)/rhow

          chav = rwork
     >         + zporos(i)*(1.-wisoi(i,k))*wsoi(i,k)*ch2o
     >         + zporos(i)*wisoi(i,k)*cice

c         Freeze liquid below tmelt
          zwsoi = min(1., wsoi(i,k))
          dh(i) = chav * (tmelt - tsoi(i,k))
          dw(i) = min(zporos(i)*(1.-wisoi(i,k))*zwsoi,
     >                max(0., dh(i)/hfus))

          wisoi(i,k) = wisoi(i,k) + dw(i)/zporos(i)
          wsoi(i,k)  = wsoi(i,k)
     >               - (dw(i)/zporos(i))*(1.-zwsoi)
     >                 / max(epsilon, 1.-wisoi(i,k))

          chav = rwork
     >         + zporos(i)*(1.-wisoi(i,k))*wsoi(i,k)*ch2o
     >         + zporos(i)*wisoi(i,k)*cice

          tsoi(i,k) = tmelt - (dh(i) - hfus*dw(i)) / chav

c         Melt ice above tmelt
          dh(i) = chav * (tsoi(i,k) - tmelt)
          dw(i) = min(zporos(i)*wisoi(i,k), max(0., dh(i)/hfus))

          wisoi(i,k) = wisoi(i,k) - dw(i)/zporos(i)
          wsoi(i,k)  = wsoi(i,k)
     >               + (dw(i)/zporos(i))
     >                 * (1.-wsoi(i,k)) / max(epsilon, 1.-wisoi(i,k))

          chav = rwork
     >         + zporos(i)*(1.-wisoi(i,k))*wsoi(i,k)*ch2o
     >         + zporos(i)*wisoi(i,k)*cice

          tsoi(i,k) = tmelt + (dh(i) - hfus*dw(i)) / chav

          porosflo(i,k) = zporos(i) * max(epsilon, 1.-wisoi(i,k))

  410   continue
  400 continue

      return
      end


c ---------------------------------------------------------------------
c puddle_freeze_thaw_block
c
c Heat-conserving puddle water phase change for the top layer.
c Carved out from soilctl (soil.f lines 487-514).
c Reads/writes wpud, wipud, tsoi(:,1) (via common /comsoi*).
c ---------------------------------------------------------------------
      subroutine puddle_freeze_thaw_block
      include 'implicit.h'
      include 'compar.h'
      include 'comsoi.h'

      integer i, k
      real chav, rwork
      real dh(npoi), dw(npoi)

      k = 1
      do 500 i = 1, npoi
        rwork = ((1.-poros(i,k))*csoi(i,k)*rhosoi(i,k)
     >          + poros(i,k)*(1.-wisoi(i,k))*wsoi(i,k)*ch2o*rhow
     >          + poros(i,k)*wisoi(i,k)*cice*rhow
     >         ) * hsoi(k)

        chav = ch2o*wpud(i) + cice*wipud(i) + rwork
        dh(i) = chav * (tmelt - tsoi(i,k))
        dw(i) = min(wpud(i), max(0., dh(i)/hfus))
        wipud(i) = wipud(i) + dw(i)
        wpud(i)  = wpud(i)  - dw(i)
        chav = ch2o*wpud(i) + cice*wipud(i) + rwork
        tsoi(i,k) = tmelt - (dh(i) - hfus*dw(i)) / chav

        dh(i) = chav * (tsoi(i,k) - tmelt)
        dw(i) = min(wipud(i), max(0., dh(i)/hfus))
        wipud(i) = wipud(i) - dw(i)
        wpud(i)  = wpud(i)  + dw(i)
        chav = ch2o*wpud(i) + cice*wipud(i) + rwork
        tsoi(i,k) = tmelt + (dh(i) - hfus*dw(i)) / chav

  500 continue

      return
      end
