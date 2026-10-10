// SGP4/SDP4 orbit propagator, compiled for both CPU and CUDA.
//
// A port of the algorithm in D. A. Vallado, P. Crawford, R. Hujsak and T. S. Kelso,
// "Revisiting Spacetrack Report #3", AIAA 2006-6753, and its reference C++ code
// (WGS-72 constants, "improved" operation mode, as the python `sgp4` package uses).
//
// Unlike the reference, initialisation and propagation are separate and propagation is
// pure: sgp4() takes a const Elset and keeps no state between calls. The reference caches
// the deep-space resonance integrator between calls; restarting it from the epoch gives
// the same result, because the integrator always takes the same 720-minute steps.
#pragma once

#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
#define KESSLER_HD __host__ __device__
#else
#define KESSLER_HD
#endif

namespace kessler {

constexpr double kPi = 3.14159265358979323846;
constexpr double kTwoPi = 2.0 * kPi;
constexpr double kDeg2Rad = kPi / 180.0;

// WGS-72 gravity constants, as used by the public TLE/OMM catalogue.
struct Grav {
    static constexpr double mu = 398600.8;            // km^3/s^2
    static constexpr double radius = 6378.135;        // km
    static constexpr double j2 = 0.001082616;
    static constexpr double j3 = -0.00000253881;
    static constexpr double j4 = -0.00000165597;
    static constexpr double j3oj2 = j3 / j2;
};

KESSLER_HD inline double xke() {
    return 60.0 / std::sqrt(Grav::radius * Grav::radius * Grav::radius / Grav::mu);
}

// Mean elements as published in a TLE or OMM.
struct MeanElements {
    double epoch;      // days since 1949-12-31 00:00 UT
    double bstar;      // 1/earth radii
    double ecco;
    double argpo;      // rad
    double inclo;      // rad
    double mo;         // rad
    double no_kozai;   // rad/min
    double nodeo;      // rad
};

// Everything sgp4() needs, precomputed once per object by sgp4init().
struct Elset {
    // inputs
    double epoch, bstar, ecco, argpo, inclo, mo, nodeo, no_unkozai;
    // near-earth
    int32_t isimp, method_deep, irez, error;
    double aycof, con41, cc1, cc4, cc5, d2, d3, d4, delmo, eta, argpdot, omgcof, sinmao,
        t2cof, t3cof, t4cof, t5cof, x1mth2, x7thm1, mdot, nodedot, xlcof, xmcof, nodecf;
    // deep-space
    double gsto;
    double e3, ee2, peo, pgho, pho, pinco, plo, se2, se3, sgh2, sgh3, sgh4, sh2, sh3, si2, si3,
        sl2, sl3, sl4, xgh2, xgh3, xgh4, xh2, xh3, xi2, xi3, xl2, xl3, xl4, zmol, zmos;
    double d2201, d2211, d3210, d3222, d4410, d4422, d5220, d5232, d5421, d5433, dedt, del1,
        del2, del3, didt, dmdt, dnodt, domdt, xfact, xlamo;
};

// Error codes, as in the reference implementation.
enum Sgp4Error : int32_t {
    kOk = 0,
    kEccentricity = 1,       // mean eccentricity out of range
    kMeanMotion = 2,         // mean motion below zero
    kPerturbedEcc = 3,       // perturbed eccentricity out of range
    kSemiLatus = 4,          // semi-latus rectum below zero
    kDecayed = 6,            // orbit has decayed
};

namespace detail {

KESSLER_HD inline double gstime(double jdut1) {
    double tut1 = (jdut1 - 2451545.0) / 36525.0;
    double temp = -6.2e-6 * tut1 * tut1 * tut1 + 0.093104 * tut1 * tut1 +
                  (876600.0 * 3600 + 8640184.812866) * tut1 + 67310.54841;
    temp = std::fmod(temp * kDeg2Rad / 240.0, kTwoPi);
    if (temp < 0.0) temp += kTwoPi;
    return temp;
}

// Lunar-solar terms shared by deep-space initialisation (reference: dscom).
struct Dscom {
    double snodm, cnodm, sinim, cosim, sinomm, cosomm, day, e3, ee2, em, emsq, gam, peo, pgho,
        pho, pinco, plo, rtemsq, se2, se3, sgh2, sgh3, sgh4, sh2, sh3, si2, si3, sl2, sl3, sl4,
        s1, s2, s3, s4, s5, s6, s7, ss1, ss2, ss3, ss4, ss5, ss6, ss7, sz1, sz2, sz3, sz11,
        sz12, sz13, sz21, sz22, sz23, sz31, sz32, sz33, xgh2, xgh3, xgh4, xh2, xh3, xi2, xi3,
        xl2, xl3, xl4, nm, z1, z2, z3, z11, z12, z13, z21, z22, z23, z31, z32, z33, zmol,
        zmos;
};

KESSLER_HD inline void dscom(double epoch, double ep, double argpp, double tc, double inclp,
                             double nodep, double np, Dscom& o) {
    const double zes = 0.01675, zel = 0.05490, c1ss = 2.9864797e-6, c1l = 4.7968065e-7,
                 zsinis = 0.39785416, zcosis = 0.91744867, zcosgs = 0.1945905,
                 zsings = -0.98088458;

    o.nm = np;
    o.em = ep;
    o.snodm = std::sin(nodep);
    o.cnodm = std::cos(nodep);
    o.sinomm = std::sin(argpp);
    o.cosomm = std::cos(argpp);
    o.sinim = std::sin(inclp);
    o.cosim = std::cos(inclp);
    o.emsq = o.em * o.em;
    double betasq = 1.0 - o.emsq;
    o.rtemsq = std::sqrt(betasq);

    o.peo = 0.0;
    o.pinco = 0.0;
    o.plo = 0.0;
    o.pgho = 0.0;
    o.pho = 0.0;
    o.day = epoch + 18261.5 + tc / 1440.0;
    double xnodce = std::fmod(4.5236020 - 9.2422029e-4 * o.day, kTwoPi);
    double stem = std::sin(xnodce);
    double ctem = std::cos(xnodce);
    double zcosil = 0.91375164 - 0.03568096 * ctem;
    double zsinil = std::sqrt(1.0 - zcosil * zcosil);
    double zsinhl = 0.089683511 * stem / zsinil;
    double zcoshl = std::sqrt(1.0 - zsinhl * zsinhl);
    o.gam = 5.8351514 + 0.0019443680 * o.day;
    double zx = 0.39785416 * stem / zsinil;
    double zy = zcoshl * ctem + 0.91744867 * zsinhl * stem;
    zx = std::atan2(zx, zy);
    zx = o.gam + zx - xnodce;
    double zcosgl = std::cos(zx);
    double zsingl = std::sin(zx);

    double zcosg = zcosgs, zsing = zsings, zcosi = zcosis, zsini = zsinis;
    double zcosh = o.cnodm, zsinh = o.snodm, cc = c1ss;
    double xnoi = 1.0 / o.nm;

    for (int lsflg = 1; lsflg <= 2; ++lsflg) {
        double a1 = zcosg * zcosh + zsing * zcosi * zsinh;
        double a3 = -zsing * zcosh + zcosg * zcosi * zsinh;
        double a7 = -zcosg * zsinh + zsing * zcosi * zcosh;
        double a8 = zsing * zsini;
        double a9 = zsing * zsinh + zcosg * zcosi * zcosh;
        double a10 = zcosg * zsini;
        double a2 = o.cosim * a7 + o.sinim * a8;
        double a4 = o.cosim * a9 + o.sinim * a10;
        double a5 = -o.sinim * a7 + o.cosim * a8;
        double a6 = -o.sinim * a9 + o.cosim * a10;

        double x1 = a1 * o.cosomm + a2 * o.sinomm;
        double x2 = a3 * o.cosomm + a4 * o.sinomm;
        double x3 = -a1 * o.sinomm + a2 * o.cosomm;
        double x4 = -a3 * o.sinomm + a4 * o.cosomm;
        double x5 = a5 * o.sinomm;
        double x6 = a6 * o.sinomm;
        double x7 = a5 * o.cosomm;
        double x8 = a6 * o.cosomm;

        o.z31 = 12.0 * x1 * x1 - 3.0 * x3 * x3;
        o.z32 = 24.0 * x1 * x2 - 6.0 * x3 * x4;
        o.z33 = 12.0 * x2 * x2 - 3.0 * x4 * x4;
        o.z1 = 3.0 * (a1 * a1 + a2 * a2) + o.z31 * o.emsq;
        o.z2 = 6.0 * (a1 * a3 + a2 * a4) + o.z32 * o.emsq;
        o.z3 = 3.0 * (a3 * a3 + a4 * a4) + o.z33 * o.emsq;
        o.z11 = -6.0 * a1 * a5 + o.emsq * (-24.0 * x1 * x7 - 6.0 * x3 * x5);
        o.z12 = -6.0 * (a1 * a6 + a3 * a5) +
                o.emsq * (-24.0 * (x2 * x7 + x1 * x8) - 6.0 * (x3 * x6 + x4 * x5));
        o.z13 = -6.0 * a3 * a6 + o.emsq * (-24.0 * x2 * x8 - 6.0 * x4 * x6);
        o.z21 = 6.0 * a2 * a5 + o.emsq * (24.0 * x1 * x5 - 6.0 * x3 * x7);
        o.z22 = 6.0 * (a4 * a5 + a2 * a6) +
                o.emsq * (24.0 * (x2 * x5 + x1 * x6) - 6.0 * (x4 * x7 + x3 * x8));
        o.z23 = 6.0 * a4 * a6 + o.emsq * (24.0 * x2 * x6 - 6.0 * x4 * x8);
        o.z1 = o.z1 + o.z1 + betasq * o.z31;
        o.z2 = o.z2 + o.z2 + betasq * o.z32;
        o.z3 = o.z3 + o.z3 + betasq * o.z33;
        o.s3 = cc * xnoi;
        o.s2 = -0.5 * o.s3 / o.rtemsq;
        o.s4 = o.s3 * o.rtemsq;
        o.s1 = -15.0 * o.em * o.s4;
        o.s5 = x1 * x3 + x2 * x4;
        o.s6 = x2 * x3 + x1 * x4;
        o.s7 = x2 * x4 - x1 * x3;

        if (lsflg == 1) {
            o.ss1 = o.s1;
            o.ss2 = o.s2;
            o.ss3 = o.s3;
            o.ss4 = o.s4;
            o.ss5 = o.s5;
            o.ss6 = o.s6;
            o.ss7 = o.s7;
            o.sz1 = o.z1;
            o.sz2 = o.z2;
            o.sz3 = o.z3;
            o.sz11 = o.z11;
            o.sz12 = o.z12;
            o.sz13 = o.z13;
            o.sz21 = o.z21;
            o.sz22 = o.z22;
            o.sz23 = o.z23;
            o.sz31 = o.z31;
            o.sz32 = o.z32;
            o.sz33 = o.z33;
            zcosg = zcosgl;
            zsing = zsingl;
            zcosi = zcosil;
            zsini = zsinil;
            zcosh = zcoshl * o.cnodm + zsinhl * o.snodm;
            zsinh = o.snodm * zcoshl - o.cnodm * zsinhl;
            cc = c1l;
        }
    }

    o.zmol = std::fmod(4.7199672 + 0.22997150 * o.day - o.gam, kTwoPi);
    o.zmos = std::fmod(6.2565837 + 0.017201977 * o.day, kTwoPi);

    o.se2 = 2.0 * o.ss1 * o.ss6;
    o.se3 = 2.0 * o.ss1 * o.ss7;
    o.si2 = 2.0 * o.ss2 * o.sz12;
    o.si3 = 2.0 * o.ss2 * (o.sz13 - o.sz11);
    o.sl2 = -2.0 * o.ss3 * o.sz2;
    o.sl3 = -2.0 * o.ss3 * (o.sz3 - o.sz1);
    o.sl4 = -2.0 * o.ss3 * (-21.0 - 9.0 * o.emsq) * zes;
    o.sgh2 = 2.0 * o.ss4 * o.sz32;
    o.sgh3 = 2.0 * o.ss4 * (o.sz33 - o.sz31);
    o.sgh4 = -18.0 * o.ss4 * zes;
    o.sh2 = -2.0 * o.ss2 * o.sz22;
    o.sh3 = -2.0 * o.ss2 * (o.sz23 - o.sz21);

    o.ee2 = 2.0 * o.s1 * o.s6;
    o.e3 = 2.0 * o.s1 * o.s7;
    o.xi2 = 2.0 * o.s2 * o.z12;
    o.xi3 = 2.0 * o.s2 * (o.z13 - o.z11);
    o.xl2 = -2.0 * o.s3 * o.z2;
    o.xl3 = -2.0 * o.s3 * (o.z3 - o.z1);
    o.xl4 = -2.0 * o.s3 * (-21.0 - 9.0 * o.emsq) * zel;
    o.xgh2 = 2.0 * o.s4 * o.z32;
    o.xgh3 = 2.0 * o.s4 * (o.z33 - o.z31);
    o.xgh4 = -18.0 * o.s4 * zel;
    o.xh2 = -2.0 * o.s2 * o.z22;
    o.xh3 = -2.0 * o.s2 * (o.z23 - o.z21);
}

// Lunar-solar periodics applied at propagation time (reference: dpper, init = 'n').
KESSLER_HD inline void dpper(const Elset& s, double t, double& ep, double& inclp, double& nodep,
                             double& argpp, double& mp) {
    const double zns = 1.19459e-5, zes = 0.01675, znl = 1.5835218e-4, zel = 0.05490;

    double zm = s.zmos + zns * t;
    double zf = zm + 2.0 * zes * std::sin(zm);
    double sinzf = std::sin(zf);
    double f2 = 0.5 * sinzf * sinzf - 0.25;
    double f3 = -0.5 * sinzf * std::cos(zf);
    double ses = s.se2 * f2 + s.se3 * f3;
    double sis = s.si2 * f2 + s.si3 * f3;
    double sls = s.sl2 * f2 + s.sl3 * f3 + s.sl4 * sinzf;
    double sghs = s.sgh2 * f2 + s.sgh3 * f3 + s.sgh4 * sinzf;
    double shs = s.sh2 * f2 + s.sh3 * f3;

    zm = s.zmol + znl * t;
    zf = zm + 2.0 * zel * std::sin(zm);
    sinzf = std::sin(zf);
    f2 = 0.5 * sinzf * sinzf - 0.25;
    f3 = -0.5 * sinzf * std::cos(zf);
    double sel = s.ee2 * f2 + s.e3 * f3;
    double sil = s.xi2 * f2 + s.xi3 * f3;
    double sll = s.xl2 * f2 + s.xl3 * f3 + s.xl4 * sinzf;
    double sghl = s.xgh2 * f2 + s.xgh3 * f3 + s.xgh4 * sinzf;
    double shll = s.xh2 * f2 + s.xh3 * f3;

    double pe = ses + sel - s.peo;
    double pinc = sis + sil - s.pinco;
    double pl = sls + sll - s.plo;
    double pgh = sghs + sghl - s.pgho;
    double ph = shs + shll - s.pho;

    inclp = inclp + pinc;
    ep = ep + pe;
    double sinip = std::sin(inclp);
    double cosip = std::cos(inclp);

    if (inclp >= 0.2) {
        ph = ph / sinip;
        pgh = pgh - cosip * ph;
        argpp = argpp + pgh;
        nodep = nodep + ph;
        mp = mp + pl;
    } else {
        // Lyddane modification for low inclination
        double sinop = std::sin(nodep);
        double cosop = std::cos(nodep);
        double alfdp = sinip * sinop;
        double betdp = sinip * cosop;
        double dalf = ph * cosop + pinc * cosip * sinop;
        double dbet = -ph * sinop + pinc * cosip * cosop;
        alfdp = alfdp + dalf;
        betdp = betdp + dbet;
        nodep = std::fmod(nodep, kTwoPi);
        double xls = mp + argpp + cosip * nodep;
        double dls = pl + pgh - pinc * nodep * sinip;
        xls = xls + dls;
        double xnoh = nodep;
        nodep = std::atan2(alfdp, betdp);
        if (std::fabs(xnoh - nodep) > kPi) {
            if (nodep < xnoh)
                nodep = nodep + kTwoPi;
            else
                nodep = nodep - kTwoPi;
        }
        mp = mp + pl;
        argpp = xls - mp - cosip * nodep;
    }
}

// Deep-space secular effects and resonance integration (reference: dspace).
KESSLER_HD inline void dspace(const Elset& s, double t, double& em, double& argpm, double& inclm,
                              double& mm, double& nodem, double& nm) {
    const double fasx2 = 0.13130908, fasx4 = 2.8843198, fasx6 = 0.37448087, g22 = 5.7686396,
                 g32 = 0.95240898, g44 = 1.8014998, g52 = 1.0508330, g54 = 4.4108898,
                 rptim = 4.37526908801129966e-3, stepp = 720.0, stepn = -720.0,
                 step2 = 259200.0;

    double theta = std::fmod(s.gsto + t * rptim, kTwoPi);
    em = em + s.dedt * t;
    inclm = inclm + s.didt * t;
    argpm = argpm + s.domdt * t;
    nodem = nodem + s.dnodt * t;
    mm = mm + s.dmdt * t;

    if (s.irez == 0) return;

    double atime = 0.0;
    double xni = s.no_unkozai;
    double xli = s.xlamo;
    double delt = (t > 0.0) ? stepp : stepn;
    double ft = 0.0, xndt = 0.0, xldot = 0.0, xnddt = 0.0;

    for (;;) {
        if (s.irez != 2) {
            xndt = s.del1 * std::sin(xli - fasx2) + s.del2 * std::sin(2.0 * (xli - fasx4)) +
                   s.del3 * std::sin(3.0 * (xli - fasx6));
            xldot = xni + s.xfact;
            xnddt = s.del1 * std::cos(xli - fasx2) + 2.0 * s.del2 * std::cos(2.0 * (xli - fasx4)) +
                    3.0 * s.del3 * std::cos(3.0 * (xli - fasx6));
            xnddt = xnddt * xldot;
        } else {
            double xomi = s.argpo + s.argpdot * atime;
            double x2omi = xomi + xomi;
            double x2li = xli + xli;
            xndt = s.d2201 * std::sin(x2omi + xli - g22) + s.d2211 * std::sin(xli - g22) +
                   s.d3210 * std::sin(xomi + xli - g32) + s.d3222 * std::sin(-xomi + xli - g32) +
                   s.d4410 * std::sin(x2omi + x2li - g44) + s.d4422 * std::sin(x2li - g44) +
                   s.d5220 * std::sin(xomi + xli - g52) + s.d5232 * std::sin(-xomi + xli - g52) +
                   s.d5421 * std::sin(xomi + x2li - g54) + s.d5433 * std::sin(-xomi + x2li - g54);
            xldot = xni + s.xfact;
            xnddt = s.d2201 * std::cos(x2omi + xli - g22) + s.d2211 * std::cos(xli - g22) +
                    s.d3210 * std::cos(xomi + xli - g32) + s.d3222 * std::cos(-xomi + xli - g32) +
                    s.d5220 * std::cos(xomi + xli - g52) + s.d5232 * std::cos(-xomi + xli - g52) +
                    2.0 * (s.d4410 * std::cos(x2omi + x2li - g44) + s.d4422 * std::cos(x2li - g44) +
                           s.d5421 * std::cos(xomi + x2li - g54) +
                           s.d5433 * std::cos(-xomi + x2li - g54));
            xnddt = xnddt * xldot;
        }

        if (std::fabs(t - atime) >= stepp) {
            xli = xli + xldot * delt + xndt * step2;
            xni = xni + xndt * delt + xnddt * step2;
            atime = atime + delt;
        } else {
            ft = t - atime;
            break;
        }
    }

    nm = xni + xndt * ft + xnddt * ft * ft * 0.5;
    double xl = xli + xldot * ft + xndt * ft * ft * 0.5;
    if (s.irez != 1)
        mm = xl - 2.0 * nodem + 2.0 * theta;
    else
        mm = xl - nodem - argpm + theta;
    double dndt = nm - s.no_unkozai;
    nm = s.no_unkozai + dndt;
}

}  // namespace detail

// Precomputes everything sgp4() needs. Check `error` afterwards.
KESSLER_HD inline Elset sgp4init(const MeanElements& m) {
    using namespace detail;
    const double temp4 = 1.5e-12;
    const double x2o3 = 2.0 / 3.0;
    const double radius = Grav::radius, j2 = Grav::j2, j4 = Grav::j4, j3oj2 = Grav::j3oj2;
    const double xk = xke();

    Elset s{};
    s.epoch = m.epoch;
    s.bstar = m.bstar;
    s.ecco = m.ecco;
    s.argpo = m.argpo;
    s.inclo = m.inclo;
    s.mo = m.mo;
    s.nodeo = m.nodeo;
    s.error = kOk;

    double ss = 78.0 / radius + 1.0;
    double qzms2ttemp = (120.0 - 78.0) / radius;
    double qzms2t = qzms2ttemp * qzms2ttemp * qzms2ttemp * qzms2ttemp;

    // initl
    double eccsq = s.ecco * s.ecco;
    double omeosq = 1.0 - eccsq;
    double rteosq = std::sqrt(omeosq);
    double cosio = std::cos(s.inclo);
    double cosio2 = cosio * cosio;
    double ak = std::pow(xk / m.no_kozai, x2o3);
    double d1 = 0.75 * j2 * (3.0 * cosio2 - 1.0) / (rteosq * omeosq);
    double del = d1 / (ak * ak);
    double adel = ak * (1.0 - del * del - del * (1.0 / 3.0 + 134.0 * del * del / 81.0));
    del = d1 / (adel * adel);
    s.no_unkozai = m.no_kozai / (1.0 + del);
    double ao = std::pow(xk / s.no_unkozai, x2o3);
    double sinio = std::sin(s.inclo);
    double po = ao * omeosq;
    double con42 = 1.0 - 5.0 * cosio2;
    s.con41 = -con42 - cosio2 - cosio2;
    double posq = po * po;
    double rp = ao * (1.0 - s.ecco);
    s.gsto = gstime(s.epoch + 2433281.5);

    if (omeosq >= 0.0 || s.no_unkozai >= 0.0) {
        s.isimp = 0;
        if (rp < 220.0 / radius + 1.0) s.isimp = 1;
        double sfour = ss;
        double qzms24 = qzms2t;
        double perige = (rp - 1.0) * radius;

        if (perige < 156.0) {
            sfour = perige - 78.0;
            if (perige < 98.0) sfour = 20.0;
            double q = (120.0 - sfour) / radius;
            qzms24 = q * q * q * q;
            sfour = sfour / radius + 1.0;
        }
        double pinvsq = 1.0 / posq;

        double tsi = 1.0 / (ao - sfour);
        s.eta = ao * s.ecco * tsi;
        double etasq = s.eta * s.eta;
        double eeta = s.ecco * s.eta;
        double psisq = std::fabs(1.0 - etasq);
        double coef = qzms24 * tsi * tsi * tsi * tsi;
        double coef1 = coef / std::pow(psisq, 3.5);
        double cc2 = coef1 * s.no_unkozai *
                     (ao * (1.0 + 1.5 * etasq + eeta * (4.0 + etasq)) +
                      0.375 * j2 * tsi / psisq * s.con41 * (8.0 + 3.0 * etasq * (8.0 + etasq)));
        s.cc1 = s.bstar * cc2;
        double cc3 = 0.0;
        if (s.ecco > 1.0e-4) cc3 = -2.0 * coef * tsi * j3oj2 * s.no_unkozai * sinio / s.ecco;
        s.x1mth2 = 1.0 - cosio2;
        s.cc4 = 2.0 * s.no_unkozai * coef1 * ao * omeosq *
                (s.eta * (2.0 + 0.5 * etasq) + s.ecco * (0.5 + 2.0 * etasq) -
                 j2 * tsi / (ao * psisq) *
                     (-3.0 * s.con41 * (1.0 - 2.0 * eeta + etasq * (1.5 - 0.5 * eeta)) +
                      0.75 * s.x1mth2 * (2.0 * etasq - eeta * (1.0 + etasq)) *
                          std::cos(2.0 * s.argpo)));
        s.cc5 = 2.0 * coef1 * ao * omeosq * (1.0 + 2.75 * (etasq + eeta) + eeta * etasq);
        double cosio4 = cosio2 * cosio2;
        double temp1 = 1.5 * j2 * pinvsq * s.no_unkozai;
        double temp2 = 0.5 * temp1 * j2 * pinvsq;
        double temp3 = -0.46875 * j4 * pinvsq * pinvsq * s.no_unkozai;
        s.mdot = s.no_unkozai + 0.5 * temp1 * rteosq * s.con41 +
                 0.0625 * temp2 * rteosq * (13.0 - 78.0 * cosio2 + 137.0 * cosio4);
        s.argpdot = -0.5 * temp1 * con42 +
                    0.0625 * temp2 * (7.0 - 114.0 * cosio2 + 395.0 * cosio4) +
                    temp3 * (3.0 - 36.0 * cosio2 + 49.0 * cosio4);
        double xhdot1 = -temp1 * cosio;
        s.nodedot = xhdot1 + (0.5 * temp2 * (4.0 - 19.0 * cosio2) +
                              2.0 * temp3 * (3.0 - 7.0 * cosio2)) *
                                 cosio;
        double xpidot = s.argpdot + s.nodedot;
        s.omgcof = s.bstar * cc3 * std::cos(s.argpo);
        s.xmcof = 0.0;
        if (s.ecco > 1.0e-4) s.xmcof = -x2o3 * coef * s.bstar / eeta;
        s.nodecf = 3.5 * omeosq * xhdot1 * s.cc1;
        s.t2cof = 1.5 * s.cc1;
        if (std::fabs(cosio + 1.0) > 1.5e-12)
            s.xlcof = -0.25 * j3oj2 * sinio * (3.0 + 5.0 * cosio) / (1.0 + cosio);
        else
            s.xlcof = -0.25 * j3oj2 * sinio * (3.0 + 5.0 * cosio) / temp4;
        s.aycof = -0.5 * j3oj2 * sinio;
        double delmotemp = 1.0 + s.eta * std::cos(s.mo);
        s.delmo = delmotemp * delmotemp * delmotemp;
        s.sinmao = std::sin(s.mo);
        s.x7thm1 = 7.0 * cosio2 - 1.0;

        // deep space: period of 225 minutes or more
        if (2.0 * kPi / s.no_unkozai >= 225.0) {
            s.method_deep = 1;
            s.isimp = 1;
            Dscom c;
            dscom(s.epoch, s.ecco, s.argpo, 0.0, s.inclo, s.nodeo, s.no_unkozai, c);
            s.e3 = c.e3;
            s.ee2 = c.ee2;
            s.peo = c.peo;
            s.pgho = c.pgho;
            s.pho = c.pho;
            s.pinco = c.pinco;
            s.plo = c.plo;
            s.se2 = c.se2;
            s.se3 = c.se3;
            s.sgh2 = c.sgh2;
            s.sgh3 = c.sgh3;
            s.sgh4 = c.sgh4;
            s.sh2 = c.sh2;
            s.sh3 = c.sh3;
            s.si2 = c.si2;
            s.si3 = c.si3;
            s.sl2 = c.sl2;
            s.sl3 = c.sl3;
            s.sl4 = c.sl4;
            s.xgh2 = c.xgh2;
            s.xgh3 = c.xgh3;
            s.xgh4 = c.xgh4;
            s.xh2 = c.xh2;
            s.xh3 = c.xh3;
            s.xi2 = c.xi2;
            s.xi3 = c.xi3;
            s.xl2 = c.xl2;
            s.xl3 = c.xl3;
            s.xl4 = c.xl4;
            s.zmol = c.zmol;
            s.zmos = c.zmos;

            // dsinit (at t = 0, so the secular rates do not move the elements yet)
            const double q22 = 1.7891679e-6, q31 = 2.1460748e-6, q33 = 2.2123015e-7,
                         root22 = 1.7891679e-6, root44 = 7.3636953e-9, root54 = 2.1765803e-9,
                         rptim = 4.37526908801129966e-3, root32 = 3.7393792e-7,
                         root52 = 1.1428639e-7, znl = 1.5835218e-4, zns = 1.19459e-5;
            double nm = c.nm, em = c.em, emsq = c.emsq, sinim = c.sinim, cosim = c.cosim;
            double inclm = s.inclo;

            s.irez = 0;
            if (nm < 0.0052359877 && nm > 0.0034906585) s.irez = 1;
            if (nm >= 8.26e-3 && nm <= 9.24e-3 && em >= 0.5) s.irez = 2;

            double ses = c.ss1 * zns * c.ss5;
            double sis = c.ss2 * zns * (c.sz11 + c.sz13);
            double sls = -zns * c.ss3 * (c.sz1 + c.sz3 - 14.0 - 6.0 * emsq);
            double sghs = c.ss4 * zns * (c.sz31 + c.sz33 - 6.0);
            double shs = -zns * c.ss2 * (c.sz21 + c.sz23);
            if (inclm < 5.2359877e-2 || inclm > kPi - 5.2359877e-2) shs = 0.0;
            if (sinim != 0.0) shs = shs / sinim;
            double sgs = sghs - cosim * shs;

            s.dedt = ses + c.s1 * znl * c.s5;
            s.didt = sis + c.s2 * znl * (c.z11 + c.z13);
            s.dmdt = sls - znl * c.s3 * (c.z1 + c.z3 - 14.0 - 6.0 * emsq);
            double sghl = c.s4 * znl * (c.z31 + c.z33 - 6.0);
            double shll = -znl * c.s2 * (c.z21 + c.z23);
            if (inclm < 5.2359877e-2 || inclm > kPi - 5.2359877e-2) shll = 0.0;
            s.domdt = sgs + sghl;
            s.dnodt = shs;
            if (sinim != 0.0) {
                s.domdt = s.domdt - cosim / sinim * shll;
                s.dnodt = s.dnodt + shll / sinim;
            }
            double theta = std::fmod(s.gsto, kTwoPi);

            if (s.irez != 0) {
                double aonv = std::pow(nm / xk, x2o3);
                if (s.irez == 2) {
                    double cosisq = cosim * cosim;
                    em = s.ecco;
                    emsq = eccsq;
                    double eoc = em * emsq;
                    double g201 = -0.306 - (em - 0.64) * 0.440;
                    double g211, g310, g322, g410, g422, g520, g533, g521, g532;
                    if (em <= 0.65) {
                        g211 = 3.616 - 13.2470 * em + 16.2900 * emsq;
                        g310 = -19.302 + 117.3900 * em - 228.4190 * emsq + 156.5910 * eoc;
                        g322 = -18.9068 + 109.7927 * em - 214.6334 * emsq + 146.5816 * eoc;
                        g410 = -41.122 + 242.6940 * em - 471.0940 * emsq + 313.9530 * eoc;
                        g422 = -146.407 + 841.8800 * em - 1629.014 * emsq + 1083.4350 * eoc;
                        g520 = -532.114 + 3017.977 * em - 5740.032 * emsq + 3708.2760 * eoc;
                    } else {
                        g211 = -72.099 + 331.819 * em - 508.738 * emsq + 266.724 * eoc;
                        g310 = -346.844 + 1582.851 * em - 2415.925 * emsq + 1246.113 * eoc;
                        g322 = -342.585 + 1554.908 * em - 2366.899 * emsq + 1215.972 * eoc;
                        g410 = -1052.797 + 4758.686 * em - 7193.992 * emsq + 3651.957 * eoc;
                        g422 = -3581.690 + 16178.110 * em - 24462.770 * emsq + 12422.520 * eoc;
                        if (em > 0.715)
                            g520 = -5149.66 + 29936.92 * em - 54087.36 * emsq + 31324.56 * eoc;
                        else
                            g520 = 1464.74 - 4664.75 * em + 3763.64 * emsq;
                    }
                    if (em < 0.7) {
                        g533 = -919.22770 + 4988.6100 * em - 9064.7700 * emsq + 5542.21 * eoc;
                        g521 = -822.71072 + 4568.6173 * em - 8491.4146 * emsq + 5337.524 * eoc;
                        g532 = -853.66600 + 4690.2500 * em - 8624.7700 * emsq + 5341.4 * eoc;
                    } else {
                        g533 = -37995.780 + 161616.52 * em - 229838.20 * emsq + 109377.94 * eoc;
                        g521 = -51752.104 + 218913.95 * em - 309468.16 * emsq + 146349.42 * eoc;
                        g532 = -40023.880 + 170470.89 * em - 242699.48 * emsq + 115605.82 * eoc;
                    }
                    double sini2 = sinim * sinim;
                    double f220 = 0.75 * (1.0 + 2.0 * cosim + cosisq);
                    double f221 = 1.5 * sini2;
                    double f321 = 1.875 * sinim * (1.0 - 2.0 * cosim - 3.0 * cosisq);
                    double f322 = -1.875 * sinim * (1.0 + 2.0 * cosim - 3.0 * cosisq);
                    double f441 = 35.0 * sini2 * f220;
                    double f442 = 39.3750 * sini2 * sini2;
                    double f522 = 9.84375 * sinim *
                                  (sini2 * (1.0 - 2.0 * cosim - 5.0 * cosisq) +
                                   0.33333333 * (-2.0 + 4.0 * cosim + 6.0 * cosisq));
                    double f523 = sinim * (4.92187512 * sini2 * (-2.0 - 4.0 * cosim + 10.0 * cosisq) +
                                           6.56250012 * (1.0 + 2.0 * cosim - 3.0 * cosisq));
                    double f542 = 29.53125 * sinim *
                                  (2.0 - 8.0 * cosim + cosisq * (-12.0 + 8.0 * cosim + 10.0 * cosisq));
                    double f543 = 29.53125 * sinim *
                                  (-2.0 - 8.0 * cosim + cosisq * (12.0 + 8.0 * cosim - 10.0 * cosisq));
                    double xno2 = nm * nm;
                    double ainv2 = aonv * aonv;
                    double temp1b = 3.0 * xno2 * ainv2;
                    double temp = temp1b * root22;
                    s.d2201 = temp * f220 * g201;
                    s.d2211 = temp * f221 * g211;
                    temp1b = temp1b * aonv;
                    temp = temp1b * root32;
                    s.d3210 = temp * f321 * g310;
                    s.d3222 = temp * f322 * g322;
                    temp1b = temp1b * aonv;
                    temp = 2.0 * temp1b * root44;
                    s.d4410 = temp * f441 * g410;
                    s.d4422 = temp * f442 * g422;
                    temp1b = temp1b * aonv;
                    temp = temp1b * root52;
                    s.d5220 = temp * f522 * g520;
                    s.d5232 = temp * f523 * g532;
                    temp = 2.0 * temp1b * root54;
                    s.d5421 = temp * f542 * g521;
                    s.d5433 = temp * f543 * g533;
                    s.xlamo = std::fmod(s.mo + s.nodeo + s.nodeo - theta - theta, kTwoPi);
                    s.xfact = s.mdot + s.dmdt + 2.0 * (s.nodedot + s.dnodt - rptim) - s.no_unkozai;
                }
                if (s.irez == 1) {
                    double g200 = 1.0 + emsq * (-2.5 + 0.8125 * emsq);
                    double g310 = 1.0 + 2.0 * emsq;
                    double g300 = 1.0 + emsq * (-6.0 + 6.60937 * emsq);
                    double f220 = 0.75 * (1.0 + cosim) * (1.0 + cosim);
                    double f311 = 0.9375 * sinim * sinim * (1.0 + 3.0 * cosim) - 0.75 * (1.0 + cosim);
                    double f330 = 1.0 + cosim;
                    f330 = 1.875 * f330 * f330 * f330;
                    s.del1 = 3.0 * nm * nm * aonv * aonv;
                    s.del2 = 2.0 * s.del1 * f220 * g200 * q22;
                    s.del3 = 3.0 * s.del1 * f330 * g300 * q33 * aonv;
                    s.del1 = s.del1 * f311 * g310 * q31 * aonv;
                    s.xlamo = std::fmod(s.mo + s.nodeo + s.argpo - theta, kTwoPi);
                    s.xfact = s.mdot + xpidot - rptim + s.dmdt + s.domdt + s.dnodt - s.no_unkozai;
                }
            }
        }

        if (s.isimp != 1) {
            double cc1sq = s.cc1 * s.cc1;
            s.d2 = 4.0 * ao * tsi * cc1sq;
            double temp = s.d2 * tsi * s.cc1 / 3.0;
            s.d3 = (17.0 * ao + sfour) * temp;
            s.d4 = 0.5 * temp * ao * tsi * (221.0 * ao + 31.0 * sfour) * s.cc1;
            s.t3cof = s.d2 + 2.0 * cc1sq;
            s.t4cof = 0.25 * (3.0 * s.d3 + s.cc1 * (12.0 * s.d2 + 10.0 * cc1sq));
            s.t5cof = 0.2 * (3.0 * s.d4 + 12.0 * s.cc1 * s.d3 + 6.0 * s.d2 * s.d2 +
                             15.0 * cc1sq * (2.0 * s.d2 + cc1sq));
        }
    }
    return s;
}

// Position (km) and velocity (km/s) in the TEME frame, `tsince` minutes after epoch.
// Returns kOk or an Sgp4Error; on error r and v are not meaningful.
KESSLER_HD inline int32_t sgp4(const Elset& s, double tsince, double r[3], double v[3]) {
    using namespace detail;
    const double temp4 = 1.5e-12;
    const double x2o3 = 2.0 / 3.0;
    const double xk = xke();
    const double vkmpersec = Grav::radius * xk / 60.0;
    const double j2 = Grav::j2, j3oj2 = Grav::j3oj2;

    double t = tsince;
    double xmdf = s.mo + s.mdot * t;
    double argpdf = s.argpo + s.argpdot * t;
    double nodedf = s.nodeo + s.nodedot * t;
    double argpm = argpdf;
    double mm = xmdf;
    double t2 = t * t;
    double nodem = nodedf + s.nodecf * t2;
    double tempa = 1.0 - s.cc1 * t;
    double tempe = s.bstar * s.cc4 * t;
    double templ = s.t2cof * t2;

    if (s.isimp != 1) {
        double delomg = s.omgcof * t;
        double delmtemp = 1.0 + s.eta * std::cos(xmdf);
        double delm = s.xmcof * (delmtemp * delmtemp * delmtemp - s.delmo);
        double temp = delomg + delm;
        mm = xmdf + temp;
        argpm = argpdf - temp;
        double t3 = t2 * t;
        double t4 = t3 * t;
        tempa = tempa - s.d2 * t2 - s.d3 * t3 - s.d4 * t4;
        tempe = tempe + s.bstar * s.cc5 * (std::sin(mm) - s.sinmao);
        templ = templ + s.t3cof * t3 + t4 * (s.t4cof + t * s.t5cof);
    }

    double nm = s.no_unkozai;
    double em = s.ecco;
    double inclm = s.inclo;
    if (s.method_deep) dspace(s, t, em, argpm, inclm, mm, nodem, nm);

    if (nm <= 0.0) return kMeanMotion;
    double am = std::pow(xk / nm, x2o3) * tempa * tempa;
    nm = xk / std::pow(am, 1.5);
    em = em - tempe;

    if (em >= 1.0 || em < -0.001) return kEccentricity;
    if (em < 1.0e-6) em = 1.0e-6;
    mm = mm + s.no_unkozai * templ;
    double xlm = mm + argpm + nodem;

    nodem = std::fmod(nodem, kTwoPi);
    argpm = std::fmod(argpm, kTwoPi);
    xlm = std::fmod(xlm, kTwoPi);
    mm = std::fmod(xlm - argpm - nodem, kTwoPi);

    double sinim = std::sin(inclm);
    double cosim = std::cos(inclm);

    double ep = em, xincp = inclm, argpp = argpm, nodep = nodem, mp = mm;
    double sinip = sinim, cosip = cosim;
    double aycof = s.aycof, xlcof = s.xlcof, con41 = s.con41, x1mth2 = s.x1mth2,
           x7thm1 = s.x7thm1;
    if (s.method_deep) {
        dpper(s, t, ep, xincp, nodep, argpp, mp);
        if (xincp < 0.0) {
            xincp = -xincp;
            nodep = nodep + kPi;
            argpp = argpp - kPi;
        }
        if (ep < 0.0 || ep > 1.0) return kPerturbedEcc;
        sinip = std::sin(xincp);
        cosip = std::cos(xincp);
        aycof = -0.5 * j3oj2 * sinip;
        if (std::fabs(cosip + 1.0) > 1.5e-12)
            xlcof = -0.25 * j3oj2 * sinip * (3.0 + 5.0 * cosip) / (1.0 + cosip);
        else
            xlcof = -0.25 * j3oj2 * sinip * (3.0 + 5.0 * cosip) / temp4;
    }

    double axnl = ep * std::cos(argpp);
    double temp = 1.0 / (am * (1.0 - ep * ep));
    double aynl = ep * std::sin(argpp) + temp * aycof;
    double xl = mp + argpp + nodep + temp * xlcof * axnl;

    // Kepler's equation
    double u = std::fmod(xl - nodep, kTwoPi);
    double eo1 = u;
    double tem5 = 9999.9;
    int ktr = 1;
    double sineo1 = 0.0, coseo1 = 0.0;
    while (std::fabs(tem5) >= 1.0e-12 && ktr <= 10) {
        sineo1 = std::sin(eo1);
        coseo1 = std::cos(eo1);
        tem5 = 1.0 - coseo1 * axnl - sineo1 * aynl;
        tem5 = (u - aynl * coseo1 + axnl * sineo1 - eo1) / tem5;
        if (std::fabs(tem5) >= 0.95) tem5 = tem5 > 0.0 ? 0.95 : -0.95;
        eo1 = eo1 + tem5;
        ktr = ktr + 1;
    }

    double ecose = axnl * coseo1 + aynl * sineo1;
    double esine = axnl * sineo1 - aynl * coseo1;
    double el2 = axnl * axnl + aynl * aynl;
    double pl = am * (1.0 - el2);
    if (pl < 0.0) return kSemiLatus;

    double rl = am * (1.0 - ecose);
    double rdotl = std::sqrt(am) * esine / rl;
    double rvdotl = std::sqrt(pl) / rl;
    double betal = std::sqrt(1.0 - el2);
    temp = esine / (1.0 + betal);
    double sinu = am / rl * (sineo1 - aynl - axnl * temp);
    double cosu = am / rl * (coseo1 - axnl + aynl * temp);
    double su = std::atan2(sinu, cosu);
    double sin2u = (cosu + cosu) * sinu;
    double cos2u = 1.0 - 2.0 * sinu * sinu;
    temp = 1.0 / pl;
    double temp1 = 0.5 * j2 * temp;
    double temp2 = temp1 * temp;

    if (s.method_deep) {
        double cosisq = cosip * cosip;
        con41 = 3.0 * cosisq - 1.0;
        x1mth2 = 1.0 - cosisq;
        x7thm1 = 7.0 * cosisq - 1.0;
    }
    double mrt = rl * (1.0 - 1.5 * temp2 * betal * con41) + 0.5 * temp1 * x1mth2 * cos2u;
    su = su - 0.25 * temp2 * x7thm1 * sin2u;
    double xnode = nodep + 1.5 * temp2 * cosip * sin2u;
    double xinc = xincp + 1.5 * temp2 * cosip * sinip * cos2u;
    double mvt = rdotl - nm * temp1 * x1mth2 * sin2u / xk;
    double rvdot = rvdotl + nm * temp1 * (x1mth2 * cos2u + 1.5 * con41) / xk;

    double sinsu = std::sin(su), cossu = std::cos(su);
    double snod = std::sin(xnode), cnod = std::cos(xnode);
    double sini = std::sin(xinc), cosi = std::cos(xinc);
    double xmx = -snod * cosi;
    double xmy = cnod * cosi;
    double ux = xmx * sinsu + cnod * cossu;
    double uy = xmy * sinsu + snod * cossu;
    double uz = sini * sinsu;
    double vx = xmx * cossu - cnod * sinsu;
    double vy = xmy * cossu - snod * sinsu;
    double vz = sini * cossu;

    r[0] = mrt * ux * Grav::radius;
    r[1] = mrt * uy * Grav::radius;
    r[2] = mrt * uz * Grav::radius;
    v[0] = (mvt * ux + rvdot * vx) * vkmpersec;
    v[1] = (mvt * uy + rvdot * vy) * vkmpersec;
    v[2] = (mvt * uz + rvdot * vz) * vkmpersec;

    if (mrt < 1.0) return kDecayed;
    return kOk;
}

}  // namespace kessler
