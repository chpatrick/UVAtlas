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
                size_t dwMaxRange,
                float epsilon=1e-10) // Spectra's default
        {
            DPF(0, "Starting SymmetricMatrix::GetEigen with dwDimension %d, dwMaxRange %d", dwDimension, dwMaxRange);

            typedef Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic> EigenMatrix;
            Eigen::Map<const EigenMatrix> matrix(pMatrix, dwDimension, dwDimension);
            Eigen::Map<EigenMatrix> eigenvalues(pEigenValue, dwMaxRange, 1);
            Eigen::Map<EigenMatrix> eigenvectors(pEigenVector, dwDimension, dwMaxRange);

            // Check arguments.
            if (!pMatrix || !pEigenValue || !pEigenVector) {
                DPF(0, "Got null pointers in SymmetricMatrix::GetEigen");
                return false;
            }

            if (dwDimension < dwMaxRange
                || dwMaxRange == 0
                || dwDimension == 0) {
                DPF(0, "Got invalid dwDimension %d, dwMaxRange %d", dwDimension, dwMaxRange);
                return false;
            }

            // If we don't want every eigenvalue, try solving with Spectra first.
            if (dwMaxRange < dwDimension) {
                DPF(0, "Using Spectra::SymEigsSolver");
                const auto spectraStartTime = std::chrono::steady_clock::now();

                constexpr int maxIterations = 1000; // Spectra's default

                // Construct matrix operation object using the wrapper class DenseSymMatProd.
                Spectra::DenseSymMatProd<value_type> op(matrix);
                // Construct eigen solver object, requesting the largest dwMaxRange eigenvalues
                Spectra::SymEigsSolver<value_type, Spectra::LARGEST_ALGE, Spectra::DenseSymMatProd<value_type> > eigs(
                    &op,
                    dwMaxRange,
                    // Convergence speed, higher is faster with more memory usage, recommended to be at least 2x nev, must be <= dimension.
                    // @chpatrick didn't find substantial differences above 2x
                    std::min(dwMaxRange * 2, dwDimension)
                );
                eigs.init();
                const int numConverged = eigs.compute(
                    maxIterations,
                    epsilon,
                    Spectra::LARGEST_ALGE // Sort by descending eigenvalues.
                );
                const auto spectraEndTime = std::chrono::steady_clock::now();

                const std::chrono::duration<double> spectraElapsed = spectraEndTime - spectraStartTime;
                DPF(0, "Spectra::SymEigsSolver took %f seconds with dwDimension %d, dwMaxRange %d", spectraElapsed.count(), dwDimension, dwMaxRange);

                if (numConverged >= dwMaxRange && eigs.info() == Spectra::SUCCESSFUL) {
                    eigenvalues = eigs.eigenvalues();
                    eigenvectors = eigs.eigenvectors();
                    return true;
                } else {
                    DPF(0, "Spectra::SymEigsSolver failed with info() == %d, numConverged == %d, dwDimension == %d, dwMaxRange == %d", eigs.info(), numConverged, dwDimension, dwMaxRange);
                }
            }

            {
                DPF(0, "Using Eigen::SelfAdjointEigenSolver");
                const auto eigenStartTime = std::chrono::steady_clock::now();
                const Eigen::SelfAdjointEigenSolver<EigenMatrix> eigenSolver(matrix);
                const auto eigenEndTime = std::chrono::steady_clock::now();

                const std::chrono::duration<double> eigenElapsed = eigenEndTime - eigenStartTime;
                DPF(0, "Eigen::SelfAdjointEigenSolver took %f seconds with dwDimension %d, dwMaxRange %d", eigenElapsed.count(), dwDimension, dwMaxRange);

                if (eigenSolver.info() == Eigen::ComputationInfo::Success) {
                    // We want the eigenvalues in descending order, Eigen produces them in increasing order.
                    eigenvalues = eigenSolver.eigenvalues().reverse().head(dwMaxRange);
                    eigenvectors = eigenSolver.eigenvectors().rowwise().reverse().leftCols(dwMaxRange);
                    return true;
                } else {
                    DPF(0, "Eigen::SelfAdjointEigenSolver failed with info() == %d", eigenSolver.info());
                }
            }

            return false;
        }
    };
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif
