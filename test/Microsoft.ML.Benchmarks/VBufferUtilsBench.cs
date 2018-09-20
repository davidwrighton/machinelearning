// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using BenchmarkDotNet.Attributes;
using System;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.KMeans;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Benchmarks
{
    public class VBufferUtilsBench
    {
        public class TestApis
        {
                    
            private struct AddMulIntoVisitor : VBufferUtils.IDstProducingPairVisitor<float, float, float>
            {
                private readonly float _c;

                public AddMulIntoVisitor(float c)
                {
                    _c = c;
                }

                public float Visit(int index, float value, float value2)
                {
                    return value + _c * value2;
                }
            }

            public static void AddMultInto_Generic(ref VBuffer<float> a, float c, ref VBuffer<float> b, ref VBuffer<float> dst)
            {
                VBufferUtils.ApplyInto(ref a, ref b, ref dst, new AddMulIntoVisitor(c));
            }

            public static void AddMultInto_Delegate(ref VBuffer<float> a, float c, ref VBuffer<float> b, ref VBuffer<float> dst)
            {
                VBufferUtils.ApplyInto(ref a, ref b, ref dst, (int index, float value, float value2) => value + c * value2);
            }

            private struct ScaleIntoVisitor : VBufferUtils.IDstProducingVisitor<float, float>
            {
                public ScaleIntoVisitor(float c)
                {
                    _c = c;
                }

                private readonly float _c;

                public float Visit(int index, float value)
                {
                    return _c * value;
                }
            }

            public static void ScaleInto_Generic(ref VBuffer<float> src, float c, ref VBuffer<float> dst)
            {
                VBufferUtils.ApplyIntoEitherDefined(ref src, ref dst, new ScaleIntoVisitor(c));
            }

            public static void ScaleInto_Delegate(ref VBuffer<float> src, float c, ref VBuffer<float> dst)
            {
                if (c == -1)
                    VBufferUtils.ApplyIntoEitherDefined(ref src, ref dst, (int index, float value) => -value);
                else
                    VBufferUtils.ApplyIntoEitherDefined(ref src, ref dst, (int index, float value) => c * value);
            }

            private struct InplaceAdder : VBufferUtils.IPairManipulator<float, float>
            {
                public void Manipulate(int slot, float v1, ref float v2) { v2 += v1; }
            }

            public static void Add_Generic(ref VBuffer<float> src, ref VBuffer<float> dst)
            {
                Contracts.Check(src.Length == dst.Length, "Vectors must have the same dimensionality.");

                if (src.Count == 0)
                    return;

                VBufferUtils.ApplyWith(ref src, ref dst, new InplaceAdder());
            }

            public static void Add_Delegate(ref VBuffer<float> src, ref VBuffer<float> dst)
            {
                Contracts.Check(src.Length == dst.Length, "Vectors must have the same dimensionality.");

                if (src.Count == 0)
                    return;

                VBufferUtils.ApplyWith(ref src, ref dst, (int slot, float v1, ref float v2) => { v2 += v1; });
            }
        }
        public enum VBufferChunkType
        {
            SparseEveryThird,
            SparseEveryOther,
            Dense
        }

        public static VBuffer<float> CreateDataVBuffer(int lengthDividedBySix, VBufferChunkType chunkType, float initialValue = 1.0f, float increment = 1.5f, int initialIndexMultipliedBy6 = 0)
        {

            int countOfValues = 0;
            int length = lengthDividedBySix * 6;
            int indexIncrement = 1;

            switch (chunkType)
            {
                case VBufferChunkType.Dense:
                    countOfValues = length;
                    if (initialIndexMultipliedBy6 != 0)
                        throw new Exception();
                    break;
                case VBufferChunkType.SparseEveryOther:
                    countOfValues = (lengthDividedBySix - initialIndexMultipliedBy6) * 3;
                    indexIncrement = 2;
                    break;
                case VBufferChunkType.SparseEveryThird:
                    countOfValues = (lengthDividedBySix - initialIndexMultipliedBy6) * 2;
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
                int currentIndexValue = initialIndexMultipliedBy6 * 6;
                for (int iIndex = 0; iIndex < data.Length; iIndex++)
                {
                    indices[iIndex] = currentIndexValue;
                    currentIndexValue += indexIncrement;
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
        private VBuffer<float> _sparseLen18000Count9000_2 = CreateDataVBuffer(3000, VBufferChunkType.SparseEveryOther, 2.0f);
        private VBuffer<float> _sparseLen18000Count3000_StartAt6000 = CreateDataVBuffer(3000, VBufferChunkType.SparseEveryOther, initialIndexMultipliedBy6:1000);
        private VBuffer<float> _sparseLen18000Count6000 = CreateDataVBuffer(3000, VBufferChunkType.SparseEveryThird);
        private VBuffer<float> _sparseLen18000Count18000 = CreateDataVBuffer(3000, VBufferChunkType.Dense);
        private VBuffer<float> _sparseLen18000Count18000_2 = CreateDataVBuffer(3000, VBufferChunkType.Dense, 2.0f);
        private VBuffer<float> _sparseLen18000Count18000_dst = CreateDataVBuffer(3000, VBufferChunkType.Dense);
        private VBuffer<float> _sparseLen54000Count18000 = CreateDataVBuffer(9000, VBufferChunkType.SparseEveryThird);

        private static VBuffer<float> _result;

        [Benchmark]
        public void ScaleInto_By4_Dense_18_Elems_Generic()
        {
            TestApis.ScaleInto_Generic(ref _sparseLen18Count18, 4.0f, ref _result);
        }
        [Benchmark]
        public void ScaleInto_By4_Sparse_18_Elems_Generic()
        {
            TestApis.ScaleInto_Generic(ref _sparseLen54Count18, 4.0f, ref _result);
        }

        [Benchmark]
        public void ScaleInto_By4_Dense_18000_Elems_Generic()
        {
            TestApis.ScaleInto_Generic(ref _sparseLen18000Count18000, 4.0f, ref _result);
        }

        [Benchmark]
        public void ScaleInto_By4_Sparse_18000_Elems_Generic()
        {
            TestApis.ScaleInto_Generic(ref _sparseLen54000Count18000, 4.0f, ref _result);
        }

        [Benchmark]
        public void ScaleInto_ByMinusOne_Dense_18_Elems_Generic()
        {
            TestApis.ScaleInto_Generic(ref _sparseLen18Count18, -1.0f, ref _result);
        }
        [Benchmark]
        public void ScaleInto_ByMinusOne_Sparse_18_Elems_Generic()
        {
            TestApis.ScaleInto_Generic(ref _sparseLen54Count18, -1.0f, ref _result);
        }

        [Benchmark]
        public void ScaleInto_ByMinusOne_Dense_18000_Elems_Generic()
        {
            TestApis.ScaleInto_Generic(ref _sparseLen18000Count18000, -1.0f, ref _result);
        }

        [Benchmark]
        public void ScaleInto_ByMinusOne_Sparse_18000_Elems_Generic()
        {
            TestApis.ScaleInto_Generic(ref _sparseLen54000Count18000, -1.0f, ref _result);
        }

        [Benchmark]
        public void AddMultInto_Dense_18_Elems_Generic()
        {
            TestApis.AddMultInto_Generic(ref _sparseLen18Count18, 4.0f, ref _sparseLen18Count18_2, ref _sparseLen18Count18_dst);
        }

        [Benchmark]
        public void AddMultInto_Dense_18000_Elems_Generic()
        {
            TestApis.AddMultInto_Generic(ref _sparseLen18000Count18000, 4.0f, ref _sparseLen18000Count18000_2, ref _sparseLen18000Count18000_dst);
        }

        [Benchmark]
        public void AddMultInto_SparseADenseB_18000_Elems_Generic()
        {
            TestApis.AddMultInto_Generic(ref _sparseLen18000Count9000, 4.0f, ref _sparseLen18000Count18000, ref _sparseLen18000Count18000_dst);
        }

        [Benchmark]
        public void AddMultInto_DenseASparseB_18000_Elems_Generic()
        {
            TestApis.AddMultInto_Generic(ref _sparseLen18000Count18000, 4.0f, ref _sparseLen18000Count9000, ref _sparseLen18000Count18000_dst);
        }

        [Benchmark]
        public void AddMultInto_SparseASparseB_SameIndices_18000_Elems_Generic()
        {
            TestApis.AddMultInto_Generic(ref _sparseLen18000Count9000, 4.0f, ref _sparseLen18000Count9000_2, ref _sparseLen18000Count18000_dst);
        }

        [Benchmark]
        public void AddMultInto_SparseASparseB_ASubsetOfIndicesOfB_18000_Elems_Generic()
        {
            TestApis.AddMultInto_Generic(ref _sparseLen18000Count3000_StartAt6000, 4.0f, ref _sparseLen18000Count9000, ref _sparseLen18000Count18000_dst);
        }

        [Benchmark]
        public void AddMultInto_SparseASparseB_BSubsetOfIndicesOfA_18000_Elems_Generic()
        {
            TestApis.AddMultInto_Generic(ref _sparseLen18000Count9000, 4.0f, ref _sparseLen18000Count3000_StartAt6000, ref _sparseLen18000Count18000_dst);
        }

        [Benchmark]
        public void AddMultInto_SparseASparseB_NeitherSubset_18000_Elems_Generic()
        {
            TestApis.AddMultInto_Generic(ref _sparseLen18000Count6000, 4.0f, ref _sparseLen18000Count3000_StartAt6000, ref _sparseLen18000Count18000_dst);
        }

        [Benchmark]
        public void AddMultInto_SparseASparseB_BSubsetOfIndicesOfA_18000_Elems_Delegate()
        {
            TestApis.AddMultInto_Delegate(ref _sparseLen18000Count9000, 4.0f, ref _sparseLen18000Count3000_StartAt6000, ref _sparseLen18000Count18000_dst);
        }

        [Benchmark]
        public void AddMultInto_Dense_18_Elems_Delegate()
        {
            TestApis.AddMultInto_Delegate(ref _sparseLen18Count18, 4.0f, ref _sparseLen18Count18_2, ref _sparseLen18Count18_dst);
        }

        [Benchmark]
        public void AddMultInto_Dense_18000_Elems_Delegate()
        {
            TestApis.AddMultInto_Delegate(ref _sparseLen18000Count18000, 4.0f, ref _sparseLen18000Count18000_2, ref _sparseLen18000Count18000_dst);
        }

        [Benchmark]
        public void ScaleInto_By4_Dense_18_Elems_Delegate()
        {
            TestApis.ScaleInto_Delegate(ref _sparseLen18Count18, 4.0f, ref _result);
        }
        [Benchmark]
        public void ScaleInto_By4_Sparse_18_Elems_Delegate()
        {
            TestApis.ScaleInto_Delegate(ref _sparseLen54Count18, 4.0f, ref _result);
        }

        [Benchmark]
        public void ScaleInto_By4_Dense_18000_Elems_Delegate()
        {
            TestApis.ScaleInto_Delegate(ref _sparseLen18000Count18000, 4.0f, ref _result);
        }

        [Benchmark]
        public void ScaleInto_By4_Sparse_18000_Elems_Delegate()
        {
            TestApis.ScaleInto_Delegate(ref _sparseLen54000Count18000, 4.0f, ref _result);
        }

        [Benchmark]
        public void ScaleInto_ByMinusOne_Dense_18_Elems_Delegate()
        {
            TestApis.ScaleInto_Delegate(ref _sparseLen18Count18, -1.0f, ref _result);
        }
        [Benchmark]
        public void ScaleInto_ByMinusOne_Sparse_18_Elems_Delegate()
        {
            TestApis.ScaleInto_Delegate(ref _sparseLen54Count18, -1.0f, ref _result);
        }

        [Benchmark]
        public void ScaleInto_ByMinusOne_Dense_18000_Elems_Delegate()
        {
            TestApis.ScaleInto_Delegate(ref _sparseLen18000Count18000, -1.0f, ref _result);
        }

        [Benchmark]
        public void ScaleInto_ByMinusOne_Sparse_18000_Elems_Delegate()
        {
            TestApis.ScaleInto_Delegate(ref _sparseLen54000Count18000, -1.0f, ref _result);
        }

        [Benchmark]
        public void Add_Dense_18000_Elems_Delegate()
        {
            TestApis.Add_Generic(ref _sparseLen18000Count18000, 4.0f, ref _sparseLen18000Count18000_2, ref _sparseLen18000Count18000_dst);
        }
    }
}