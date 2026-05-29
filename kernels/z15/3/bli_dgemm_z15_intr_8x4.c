/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2026, IBM Corp.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "blis.h"

#include <vecintrin.h>

static void bli_z15_dgemm_store_scalar_8x4
     (
             dim_t  m,
             dim_t  n,
       const double alpha,
       const __vector double   ab0[4],
       const __vector double   ab1[4],
       const __vector double   ab2[4],
       const __vector double   ab3[4],
       const double beta,
             double* c, inc_t rs_c, inc_t cs_c
     )
{
	double ab_scalar[8][4];

	for ( dim_t j = 0; j < 4; ++j )
	{
		ab_scalar[0][j] = ab0[j][0];
		ab_scalar[1][j] = ab0[j][1];
		ab_scalar[2][j] = ab1[j][0];
		ab_scalar[3][j] = ab1[j][1];
		ab_scalar[4][j] = ab2[j][0];
		ab_scalar[5][j] = ab2[j][1];
		ab_scalar[6][j] = ab3[j][0];
		ab_scalar[7][j] = ab3[j][1];
	}

	for ( dim_t j = 0; j < n; ++j )
	{
		for ( dim_t i = 0; i < m; ++i )
		{
			double* cij = c + i*rs_c + j*cs_c;
			const double ab = ab_scalar[i][j];

			if ( beta == 0.0 )
				*cij = alpha * ab;
			else
				*cij = alpha * ab + beta * *cij;
		}
	}
}

static void bli_z15_dgemm_8x4
     (
       const dim_t   k,
       const double  alpha,
       const double* restrict a,
       const double* restrict b,
       const double  beta,
             double* restrict c, const inc_t cs_c
     )
{
	__vector double c00 = vec_splats( 0.0 );
	__vector double c10 = vec_splats( 0.0 );
	__vector double c20 = vec_splats( 0.0 );
	__vector double c30 = vec_splats( 0.0 );
	__vector double c01 = vec_splats( 0.0 );
	__vector double c11 = vec_splats( 0.0 );
	__vector double c21 = vec_splats( 0.0 );
	__vector double c31 = vec_splats( 0.0 );
	__vector double c02 = vec_splats( 0.0 );
	__vector double c12 = vec_splats( 0.0 );
	__vector double c22 = vec_splats( 0.0 );
	__vector double c32 = vec_splats( 0.0 );
	__vector double c03 = vec_splats( 0.0 );
	__vector double c13 = vec_splats( 0.0 );
	__vector double c23 = vec_splats( 0.0 );
	__vector double c33 = vec_splats( 0.0 );

	dim_t p = 0;
	for ( ; p + 3 < k; p += 4 )
	{
		const double* restrict a_p0 = a + p*8;
		const double* restrict b_p0 = b + p*4;
		const double* restrict a_p1 = a_p0 + 8;
		const double* restrict b_p1 = b_p0 + 4;
		const double* restrict a_p2 = a_p1 + 8;
		const double* restrict b_p2 = b_p1 + 4;
		const double* restrict a_p3 = a_p2 + 8;
		const double* restrict b_p3 = b_p2 + 4;

		if ( p + 7 < k )
		{
			__builtin_prefetch( a + (p+4)*8, 0, 3 );
			__builtin_prefetch( b + (p+4)*4, 0, 3 );
		}

		const __vector double a0_0 = vec_xl( 0, a_p0 + 0 );
		const __vector double a1_0 = vec_xl( 0, a_p0 + 2 );
		const __vector double a2_0 = vec_xl( 0, a_p0 + 4 );
		const __vector double a3_0 = vec_xl( 0, a_p0 + 6 );
		const __vector double b0_0 = vec_splats( b_p0[0] );
		const __vector double b1_0 = vec_splats( b_p0[1] );
		const __vector double b2_0 = vec_splats( b_p0[2] );
		const __vector double b3_0 = vec_splats( b_p0[3] );

		c00 = vec_madd( a0_0, b0_0, c00 );
		c01 = vec_madd( a0_0, b1_0, c01 );
		c02 = vec_madd( a0_0, b2_0, c02 );
		c03 = vec_madd( a0_0, b3_0, c03 );

		c10 = vec_madd( a1_0, b0_0, c10 );
		c11 = vec_madd( a1_0, b1_0, c11 );
		c12 = vec_madd( a1_0, b2_0, c12 );
		c13 = vec_madd( a1_0, b3_0, c13 );

		c20 = vec_madd( a2_0, b0_0, c20 );
		c21 = vec_madd( a2_0, b1_0, c21 );
		c22 = vec_madd( a2_0, b2_0, c22 );
		c23 = vec_madd( a2_0, b3_0, c23 );

		c30 = vec_madd( a3_0, b0_0, c30 );
		c31 = vec_madd( a3_0, b1_0, c31 );
		c32 = vec_madd( a3_0, b2_0, c32 );
		c33 = vec_madd( a3_0, b3_0, c33 );

		const __vector double a0_1 = vec_xl( 0, a_p1 + 0 );
		const __vector double a1_1 = vec_xl( 0, a_p1 + 2 );
		const __vector double a2_1 = vec_xl( 0, a_p1 + 4 );
		const __vector double a3_1 = vec_xl( 0, a_p1 + 6 );
		const __vector double b0_1 = vec_splats( b_p1[0] );
		const __vector double b1_1 = vec_splats( b_p1[1] );
		const __vector double b2_1 = vec_splats( b_p1[2] );
		const __vector double b3_1 = vec_splats( b_p1[3] );

		c00 = vec_madd( a0_1, b0_1, c00 );
		c01 = vec_madd( a0_1, b1_1, c01 );
		c02 = vec_madd( a0_1, b2_1, c02 );
		c03 = vec_madd( a0_1, b3_1, c03 );

		c10 = vec_madd( a1_1, b0_1, c10 );
		c11 = vec_madd( a1_1, b1_1, c11 );
		c12 = vec_madd( a1_1, b2_1, c12 );
		c13 = vec_madd( a1_1, b3_1, c13 );

		c20 = vec_madd( a2_1, b0_1, c20 );
		c21 = vec_madd( a2_1, b1_1, c21 );
		c22 = vec_madd( a2_1, b2_1, c22 );
		c23 = vec_madd( a2_1, b3_1, c23 );

		c30 = vec_madd( a3_1, b0_1, c30 );
		c31 = vec_madd( a3_1, b1_1, c31 );
		c32 = vec_madd( a3_1, b2_1, c32 );
		c33 = vec_madd( a3_1, b3_1, c33 );

		const __vector double a0_2 = vec_xl( 0, a_p2 + 0 );
		const __vector double a1_2 = vec_xl( 0, a_p2 + 2 );
		const __vector double a2_2 = vec_xl( 0, a_p2 + 4 );
		const __vector double a3_2 = vec_xl( 0, a_p2 + 6 );
		const __vector double b0_2 = vec_splats( b_p2[0] );
		const __vector double b1_2 = vec_splats( b_p2[1] );
		const __vector double b2_2 = vec_splats( b_p2[2] );
		const __vector double b3_2 = vec_splats( b_p2[3] );

		c00 = vec_madd( a0_2, b0_2, c00 );
		c01 = vec_madd( a0_2, b1_2, c01 );
		c02 = vec_madd( a0_2, b2_2, c02 );
		c03 = vec_madd( a0_2, b3_2, c03 );

		c10 = vec_madd( a1_2, b0_2, c10 );
		c11 = vec_madd( a1_2, b1_2, c11 );
		c12 = vec_madd( a1_2, b2_2, c12 );
		c13 = vec_madd( a1_2, b3_2, c13 );

		c20 = vec_madd( a2_2, b0_2, c20 );
		c21 = vec_madd( a2_2, b1_2, c21 );
		c22 = vec_madd( a2_2, b2_2, c22 );
		c23 = vec_madd( a2_2, b3_2, c23 );

		c30 = vec_madd( a3_2, b0_2, c30 );
		c31 = vec_madd( a3_2, b1_2, c31 );
		c32 = vec_madd( a3_2, b2_2, c32 );
		c33 = vec_madd( a3_2, b3_2, c33 );

		const __vector double a0_3 = vec_xl( 0, a_p3 + 0 );
		const __vector double a1_3 = vec_xl( 0, a_p3 + 2 );
		const __vector double a2_3 = vec_xl( 0, a_p3 + 4 );
		const __vector double a3_3 = vec_xl( 0, a_p3 + 6 );
		const __vector double b0_3 = vec_splats( b_p3[0] );
		const __vector double b1_3 = vec_splats( b_p3[1] );
		const __vector double b2_3 = vec_splats( b_p3[2] );
		const __vector double b3_3 = vec_splats( b_p3[3] );

		c00 = vec_madd( a0_3, b0_3, c00 );
		c01 = vec_madd( a0_3, b1_3, c01 );
		c02 = vec_madd( a0_3, b2_3, c02 );
		c03 = vec_madd( a0_3, b3_3, c03 );

		c10 = vec_madd( a1_3, b0_3, c10 );
		c11 = vec_madd( a1_3, b1_3, c11 );
		c12 = vec_madd( a1_3, b2_3, c12 );
		c13 = vec_madd( a1_3, b3_3, c13 );

		c20 = vec_madd( a2_3, b0_3, c20 );
		c21 = vec_madd( a2_3, b1_3, c21 );
		c22 = vec_madd( a2_3, b2_3, c22 );
		c23 = vec_madd( a2_3, b3_3, c23 );

		c30 = vec_madd( a3_3, b0_3, c30 );
		c31 = vec_madd( a3_3, b1_3, c31 );
		c32 = vec_madd( a3_3, b2_3, c32 );
		c33 = vec_madd( a3_3, b3_3, c33 );
	}

	for ( ; p < k; ++p )
	{
		const double* restrict a_p = a + p*8;
		const double* restrict b_p = b + p*4;

		const __vector double a0 = vec_xl( 0, a_p + 0 );
		const __vector double a1 = vec_xl( 0, a_p + 2 );
		const __vector double a2 = vec_xl( 0, a_p + 4 );
		const __vector double a3 = vec_xl( 0, a_p + 6 );
		const __vector double b0 = vec_splats( b_p[0] );
		const __vector double b1 = vec_splats( b_p[1] );
		const __vector double b2 = vec_splats( b_p[2] );
		const __vector double b3 = vec_splats( b_p[3] );

		c00 = vec_madd( a0, b0, c00 );
		c01 = vec_madd( a0, b1, c01 );
		c02 = vec_madd( a0, b2, c02 );
		c03 = vec_madd( a0, b3, c03 );

		c10 = vec_madd( a1, b0, c10 );
		c11 = vec_madd( a1, b1, c11 );
		c12 = vec_madd( a1, b2, c12 );
		c13 = vec_madd( a1, b3, c13 );

		c20 = vec_madd( a2, b0, c20 );
		c21 = vec_madd( a2, b1, c21 );
		c22 = vec_madd( a2, b2, c22 );
		c23 = vec_madd( a2, b3, c23 );

		c30 = vec_madd( a3, b0, c30 );
		c31 = vec_madd( a3, b1, c31 );
		c32 = vec_madd( a3, b2, c32 );
		c33 = vec_madd( a3, b3, c33 );
	}

	if ( __builtin_expect( alpha != 1.0, 0 ) )
	{
		const __vector double alpha_v = vec_splats( alpha );

		c00 *= alpha_v;
		c01 *= alpha_v;
		c02 *= alpha_v;
		c03 *= alpha_v;

		c10 *= alpha_v;
		c11 *= alpha_v;
		c12 *= alpha_v;
		c13 *= alpha_v;

		c20 *= alpha_v;
		c21 *= alpha_v;
		c22 *= alpha_v;
		c23 *= alpha_v;

		c30 *= alpha_v;
		c31 *= alpha_v;
		c32 *= alpha_v;
		c33 *= alpha_v;
	}

	double* restrict c0 = c;
	double* restrict c1 = c0 + cs_c;
	double* restrict c2 = c1 + cs_c;
	double* restrict c3 = c2 + cs_c;

	if ( __builtin_expect( beta != 0.0, 0 ) )
	{
		const __vector double beta_v = vec_splats( beta );

		c00 = vec_madd( beta_v, vec_xl( 0, c0 + 0 ), c00 );
		c10 = vec_madd( beta_v, vec_xl( 0, c0 + 2 ), c10 );
		c20 = vec_madd( beta_v, vec_xl( 0, c0 + 4 ), c20 );
		c30 = vec_madd( beta_v, vec_xl( 0, c0 + 6 ), c30 );

		c01 = vec_madd( beta_v, vec_xl( 0, c1 + 0 ), c01 );
		c11 = vec_madd( beta_v, vec_xl( 0, c1 + 2 ), c11 );
		c21 = vec_madd( beta_v, vec_xl( 0, c1 + 4 ), c21 );
		c31 = vec_madd( beta_v, vec_xl( 0, c1 + 6 ), c31 );

		c02 = vec_madd( beta_v, vec_xl( 0, c2 + 0 ), c02 );
		c12 = vec_madd( beta_v, vec_xl( 0, c2 + 2 ), c12 );
		c22 = vec_madd( beta_v, vec_xl( 0, c2 + 4 ), c22 );
		c32 = vec_madd( beta_v, vec_xl( 0, c2 + 6 ), c32 );

		c03 = vec_madd( beta_v, vec_xl( 0, c3 + 0 ), c03 );
		c13 = vec_madd( beta_v, vec_xl( 0, c3 + 2 ), c13 );
		c23 = vec_madd( beta_v, vec_xl( 0, c3 + 4 ), c23 );
		c33 = vec_madd( beta_v, vec_xl( 0, c3 + 6 ), c33 );
	}

	vec_xst( c00, 0, c0 + 0 );
	vec_xst( c10, 0, c0 + 2 );
	vec_xst( c20, 0, c0 + 4 );
	vec_xst( c30, 0, c0 + 6 );
	vec_xst( c01, 0, c1 + 0 );
	vec_xst( c11, 0, c1 + 2 );
	vec_xst( c21, 0, c1 + 4 );
	vec_xst( c31, 0, c1 + 6 );
	vec_xst( c02, 0, c2 + 0 );
	vec_xst( c12, 0, c2 + 2 );
	vec_xst( c22, 0, c2 + 4 );
	vec_xst( c32, 0, c2 + 6 );
	vec_xst( c03, 0, c3 + 0 );
	vec_xst( c13, 0, c3 + 2 );
	vec_xst( c23, 0, c3 + 4 );
	vec_xst( c33, 0, c3 + 6 );
}

void bli_dgemm_z15_intr_8x4
     (
             dim_t      m,
             dim_t      n,
             dim_t      k,
       const void*      alpha,
       const void*      a,
       const void*      b,
       const void*      beta,
             void*      c, inc_t rs_c, inc_t cs_c,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
	const double* restrict alpha_cast = alpha;
	const double* restrict a_cast     = a;
	const double* restrict b_cast     = b;
	const double* restrict beta_cast  = beta;
	      double* restrict c_cast     = c;

	const double alpha_val = *alpha_cast;
	const double beta_val  = *beta_cast;

	( void )data;
	( void )cntx;

	if ( m < 8 || n < 4 || rs_c != 1 )
	{
		__vector double ab0[4] = { vec_splats( 0.0 ),
		                           vec_splats( 0.0 ),
		                           vec_splats( 0.0 ),
		                           vec_splats( 0.0 ) };
		__vector double ab1[4] = { vec_splats( 0.0 ),
		                           vec_splats( 0.0 ),
		                           vec_splats( 0.0 ),
		                           vec_splats( 0.0 ) };
		__vector double ab2[4] = { vec_splats( 0.0 ),
		                           vec_splats( 0.0 ),
		                           vec_splats( 0.0 ),
		                           vec_splats( 0.0 ) };
		__vector double ab3[4] = { vec_splats( 0.0 ),
		                           vec_splats( 0.0 ),
		                           vec_splats( 0.0 ),
		                           vec_splats( 0.0 ) };

		for ( dim_t p = 0; p < k; ++p )
		{
			const double* restrict a_p = a_cast + p*8;
			const double* restrict b_p = b_cast + p*4;

			const __vector double a0 = vec_xl( 0, a_p + 0 );
			const __vector double a1 = vec_xl( 0, a_p + 2 );
			const __vector double a2 = vec_xl( 0, a_p + 4 );
			const __vector double a3 = vec_xl( 0, a_p + 6 );
			const __vector double b0 = vec_splats( b_p[0] );
			const __vector double b1 = vec_splats( b_p[1] );
			const __vector double b2 = vec_splats( b_p[2] );
			const __vector double b3 = vec_splats( b_p[3] );

			ab0[0] = vec_madd( a0, b0, ab0[0] );
			ab0[1] = vec_madd( a0, b1, ab0[1] );
			ab0[2] = vec_madd( a0, b2, ab0[2] );
			ab0[3] = vec_madd( a0, b3, ab0[3] );

			ab1[0] = vec_madd( a1, b0, ab1[0] );
			ab1[1] = vec_madd( a1, b1, ab1[1] );
			ab1[2] = vec_madd( a1, b2, ab1[2] );
			ab1[3] = vec_madd( a1, b3, ab1[3] );

			ab2[0] = vec_madd( a2, b0, ab2[0] );
			ab2[1] = vec_madd( a2, b1, ab2[1] );
			ab2[2] = vec_madd( a2, b2, ab2[2] );
			ab2[3] = vec_madd( a2, b3, ab2[3] );

			ab3[0] = vec_madd( a3, b0, ab3[0] );
			ab3[1] = vec_madd( a3, b1, ab3[1] );
			ab3[2] = vec_madd( a3, b2, ab3[2] );
			ab3[3] = vec_madd( a3, b3, ab3[3] );
		}

		bli_z15_dgemm_store_scalar_8x4
		(
		  m,
		  n,
		  alpha_val,
		  ab0,
		  ab1,
		  ab2,
		  ab3,
		  beta_val,
		  c_cast, rs_c, cs_c
		);

		return;
	}

	bli_z15_dgemm_8x4
	(
	  k,
	  alpha_val,
	  a_cast,
	  b_cast,
	  beta_val,
	  c_cast, cs_c
	);
}
