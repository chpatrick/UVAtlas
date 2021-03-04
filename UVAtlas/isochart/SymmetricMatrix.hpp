//-------------------------------------------------------------------------------------
// UVAtlas - SymmetricMatrix.hpp
//
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// http://go.microsoft.com/fwlink/?LinkID=512686
//-------------------------------------------------------------------------------------

// This file used to implement the algorithm in "Numerical Recipes in Fortan 77,
// The Art of Scientific Computing Second Edition", Section 11.1 ~ 11.3
// http://www.library.cornell.edu/nr/bookfpdf/f11-1.pdf
// http://www.library.cornell.edu/nr/bookfpdf/f11-2.pdf
// http://www.library.cornell.edu/nr/bookfpdf/f11-3.pdf

#pragma once

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdouble-promotion"
#endif

namespace Isochart
{
    template<class TYPE>
    class CSymmetricMatrix
    {
    public:
        typedef TYPE value_type;

    private:

    public:
        _Success_(return)
            static bool
            GetEigen(
                size_t dwDimension,
                _In_reads_(dwDimension* dwDimension) const value_type* pMatrix,
                _Out_writes_(dwMaxRange) value_type* pEigenValue,
                _Out_writes_(dwDimension* dwMaxRange) value_type* pEigenVector,
                size_t dwMaxRange)
        {
            // Check arguments.
            if (!pMatrix || !pEigenValue || !pEigenVector) {
                return false;
            }

            if (dwDimension < dwMaxRange
                || dwMaxRange == 0
                || dwDimension == 0)
            {
                return false;
            }

            typedef Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic> EigenMatrix;
            Eigen::Map<const EigenMatrix> matrix(pMatrix, dwDimension, dwDimension);
            Eigen::Map<EigenMatrix> eigenvalues(pEigenValue, dwMaxRange, 1);
            Eigen::Map<EigenMatrix> eigenvectors(pEigenVector, dwDimension, dwMaxRange);

            Eigen::SelfAdjointEigenSolver<EigenMatrix> eigenSolver(matrix);
            if (eigenSolver.info() != Eigen::ComputationInfo::Success) {
                return false;
            }

            // Select the dwMaxRange largest eigenvalues and corresponding eigenvectors in descending order. Eigen sorts them in increasing order.
            eigenvalues = eigenSolver.eigenvalues().reverse().head(dwMaxRange);
            eigenvectors = eigenSolver.eigenvectors().rowwise().reverse().leftCols(dwMaxRange);

            return true;
        }
    };
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif
