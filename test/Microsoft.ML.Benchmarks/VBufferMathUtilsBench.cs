﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using BenchmarkDotNet.Attributes;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.KMeans;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Numeric;

namespace Microsoft.ML.Benchmarks
{
    public class VBufferMathUtilsBench
    {
        public enum VBufferChunkType
        {
            SparseEveryThird,
            SparseEveryOther,
            Dense
        }

        public static VBuffer<float> CreateDataVBuffer(int lengthDividedBySix, VBufferChunkType chunkType, float initialValue = 1.0f, float increment = 1.5f)
        {

            int countOfValues = 0;
            int length = lengthDividedBySix * 6;
            int indexIncrement = 1;

            switch (chunkType)
            {
                case VBufferChunkType.Dense:
                    countOfValues = length;
                    break;
                case VBufferChunkType.SparseEveryOther:
                    countOfValues = lengthDividedBySix * 3;
                    indexIncrement = 2;
                    break;
                case VBufferChunkType.SparseEveryThird:
                    countOfValues = lengthDividedBySix * 2;
                    indexIncrement = 3;
                    break;
            }

            float []data = new float[countOfValues];
            float currentValue = initialValue;
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = currentValue;
                currentValue += increment;
            }

            int[] indices = null;
            if (indexIncrement != 1)
            {
                indices = new int[countOfValues];
                int currentIndexValue = 0;
                for (int iIndex = 0; iIndex < data.Length; iIndex++)
                {
                    indices[iIndex] = currentIndexValue;
                    currentValue += indexIncrement;
                }
            }

            return new VBuffer<float>(length, countOfValues, data, indices);
        }

        private VBuffer<float> _sparseLen18Count9 = CreateDataVBuffer(3, VBufferChunkType.SparseEveryOther);
        private VBuffer<float> _sparseLen18Count6 = CreateDataVBuffer(3, VBufferChunkType.SparseEveryThird);
        private VBuffer<float> _sparseLen18Count18 = CreateDataVBuffer(3, VBufferChunkType.Dense);
        private VBuffer<float> _sparseLen18Count18_2 = CreateDataVBuffer(3, VBufferChunkType.Dense, 2.0f);
        private VBuffer<float> _sparseLen18Count18_dst = CreateDataVBuffer(3, VBufferChunkType.Dense);
        private VBuffer<float> _sparseLen54Count18 = CreateDataVBuffer(9, VBufferChunkType.SparseEveryThird);

        private VBuffer<float> _sparseLen180Count90 = CreateDataVBuffer(30, VBufferChunkType.SparseEveryOther);
        private VBuffer<float> _sparseLen180Count60 = CreateDataVBuffer(30, VBufferChunkType.SparseEveryThird);
        private VBuffer<float> _sparseLen180Count180 = CreateDataVBuffer(30, VBufferChunkType.Dense);
        private VBuffer<float> _sparseLen540Count180 = CreateDataVBuffer(90, VBufferChunkType.SparseEveryThird);
        
        private VBuffer<float> _sparseLen1800Count900 = CreateDataVBuffer(300, VBufferChunkType.SparseEveryOther);
        private VBuffer<float> _sparseLen1800Count600 = CreateDataVBuffer(300, VBufferChunkType.SparseEveryThird);
        private VBuffer<float> _sparseLen1800Count1800 = CreateDataVBuffer(300, VBufferChunkType.Dense);
        private VBuffer<float> _sparseLen5400Count1800 = CreateDataVBuffer(900, VBufferChunkType.SparseEveryThird);

        private VBuffer<float> _sparseLen18000Count9000 = CreateDataVBuffer(3000, VBufferChunkType.SparseEveryOther);
        private VBuffer<float> _sparseLen18000Count6000 = CreateDataVBuffer(3000, VBufferChunkType.SparseEveryThird);
        private VBuffer<float> _sparseLen18000Count18000 = CreateDataVBuffer(3000, VBufferChunkType.Dense);
        private VBuffer<float> _sparseLen18000Count18000_2 = CreateDataVBuffer(3000, VBufferChunkType.Dense, 2.0f);
        private VBuffer<float> _sparseLen18000Count18000_dst = CreateDataVBuffer(3000, VBufferChunkType.Dense);
        private VBuffer<float> _sparseLen54000Count18000 = CreateDataVBuffer(9000, VBufferChunkType.SparseEveryThird);

        private static VBuffer<float> _result;

        [Benchmark]
        public void ScaleInto_By4_Dense_18_Elems()
        {
            VectorUtils.ScaleInto(ref _sparseLen18Count18, 4.0f, ref _result);
        }
        [Benchmark]
        public void ScaleInto_By4_Sparse_18_Elems()
        {
            VectorUtils.ScaleInto(ref _sparseLen54Count18, 4.0f, ref _result);
        }

        [Benchmark]
        public void ScaleInto_By4_Dense_18000_Elems()
        {
            VectorUtils.ScaleInto(ref _sparseLen18000Count18000, 4.0f, ref _result);
        }

        [Benchmark]
        public void ScaleInto_By4_Sparse_18000_Elems()
        {
            VectorUtils.ScaleInto(ref _sparseLen54000Count18000, 4.0f, ref _result);
        }

        [Benchmark]
        public void ScaleInto_ByMinusOne_Dense_18_Elems()
        {
            VectorUtils.ScaleInto(ref _sparseLen18Count18, -1.0f, ref _result);
        }
        [Benchmark]
        public void ScaleInto_ByMinusOne_Sparse_18_Elems()
        {
            VectorUtils.ScaleInto(ref _sparseLen54Count18, -1.0f, ref _result);
        }

        [Benchmark]
        public void ScaleInto_ByMinusOne_Dense_18000_Elems()
        {
            VectorUtils.ScaleInto(ref _sparseLen18000Count18000, -1.0f, ref _result);
        }

        [Benchmark]
        public void ScaleInto_ByMinusOne_Sparse_18000_Elems()
        {
            VectorUtils.ScaleInto(ref _sparseLen54000Count18000, -1.0f, ref _result);
        }

        [Benchmark]
        public void AddMultInto_Dense_18_Elems()
        {
            VectorUtils.AddMultInto(ref _sparseLen18Count18, 4.0f, ref _sparseLen18Count18_2, ref _sparseLen18Count18_dst);
        }

        [Benchmark]
        public void AddMultInto_Dense_18000_Elems()
        {
            VectorUtils.AddMultInto(ref _sparseLen18000Count18000, 4.0f, ref _sparseLen18000Count18000_2, ref _sparseLen18000Count18000_dst);
        }
    }
}