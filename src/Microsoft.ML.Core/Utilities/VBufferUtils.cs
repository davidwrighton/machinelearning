// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#define DELEGATE_BASED_VBUFFER_UTILS
using System;
using System.Collections.Generic;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Runtime.Internal.Utilities
{
    // REVIEW: Consider automatic densification in some of the operations, where appropriate.
    // REVIEW: Once we do the conversions from Vector/WritableVector, review names of methods,
    //   parameters, parameter order, etc.
    /// <summary>
    /// Convenience utilities for vector operations on <see cref="VBuffer{T}"/>.
    /// </summary>
    public static class VBufferUtils
    {
        private const float SparsityThreshold = 0.25f;

        /// <summary>
        /// A helper method that gives us an iterable over the items given the fields from a <see cref="VBuffer{T}"/>.
        /// Note that we have this in a separate utility class, rather than in its more natural location of
        /// <see cref="VBuffer{T}"/> itself, due to a bug in the C++/CLI compiler. (DevDiv 1097919:
        /// [C++/CLI] Nested generic types are not correctly imported from metadata). So, if we want to use
        /// <see cref="VBuffer{T}"/> in C++/CLI projects, we cannot have a generic struct with a nested class
        /// that has the outer struct type as a field.
        /// </summary>
        internal static IEnumerable<KeyValuePair<int, T>> Items<T>(T[] values, int[] indices, int length, int count, bool all)
        {
            Contracts.AssertValueOrNull(values);
            Contracts.Assert(0 <= count && count <= Utils.Size(values));
            Contracts.Assert(count <= length);
            Contracts.Assert(count == length || count <= Utils.Size(indices));

            if (count == length)
            {
                for (int i = 0; i < count; i++)
                    yield return new KeyValuePair<int, T>(i, values[i]);
            }
            else if (!all)
            {
                for (int i = 0; i < count; i++)
                    yield return new KeyValuePair<int, T>(indices[i], values[i]);
            }
            else
            {
                int slotCur = -1;
                for (int i = 0; i < count; i++)
                {
                    int slot = indices[i];
                    Contracts.Assert(slotCur < slot && slot < length);
                    while (++slotCur < slot)
                        yield return new KeyValuePair<int, T>(slotCur, default(T));
                    Contracts.Assert(slotCur == slot);
                    yield return new KeyValuePair<int, T>(slotCur, values[i]);
                }
                Contracts.Assert(slotCur < length);
                while (++slotCur < length)
                    yield return new KeyValuePair<int, T>(slotCur, default(T));
            }
        }

        internal static IEnumerable<T> DenseValues<T>(T[] values, int[] indices, int length, int count)
        {
            Contracts.AssertValueOrNull(values);
            Contracts.Assert(0 <= count && count <= Utils.Size(values));
            Contracts.Assert(count <= length);
            Contracts.Assert(count == length || count <= Utils.Size(indices));

            if (count == length)
            {
                for (int i = 0; i < length; i++)
                    yield return values[i];
            }
            else
            {
                int slotCur = -1;
                for (int i = 0; i < count; i++)
                {
                    int slot = indices[i];
                    Contracts.Assert(slotCur < slot && slot < length);
                    while (++slotCur < slot)
                        yield return default(T);
                    Contracts.Assert(slotCur == slot);
                    yield return values[i];
                }
                Contracts.Assert(slotCur < length);
                while (++slotCur < length)
                    yield return default(T);
            }
        }

        public static bool HasNaNs(ref VBuffer<Single> buffer)
        {
            for (int i = 0; i < buffer.Count; i++)
            {
                if (Single.IsNaN(buffer.Values[i]))
                    return true;
            }
            return false;
        }

        public static bool HasNaNs(ref VBuffer<Double> buffer)
        {
            for (int i = 0; i < buffer.Count; i++)
            {
                if (Double.IsNaN(buffer.Values[i]))
                    return true;
            }
            return false;
        }

        public static bool HasNonFinite(ref VBuffer<Single> buffer)
        {
            for (int i = 0; i < buffer.Count; i++)
            {
                if (!FloatUtils.IsFinite(buffer.Values[i]))
                    return true;
            }
            return false;
        }

        public static bool HasNonFinite(ref VBuffer<Double> buffer)
        {
            for (int i = 0; i < buffer.Count; i++)
            {
                if (!FloatUtils.IsFinite(buffer.Values[i]))
                    return true;
            }
            return false;
        }

        public static VBuffer<T> CreateEmpty<T>(int length)
        {
            Contracts.CheckParam(length >= 0, nameof(length));
            return new VBuffer<T>(length, 0, null, null);
        }

        public static VBuffer<T> CreateDense<T>(int length)
        {
            Contracts.CheckParam(length >= 0, nameof(length));
            return new VBuffer<T>(length, new T[length]);
        }

        public interface IForEachDefinedVisitor<T>
        {
            void Visit(int index, T value);
        }

        public interface IForEachDefinedWithContextVisitor<T, TContext>
        {
            void Visit(int index, T value, ref TContext context);
        }

        private struct ForEachDefinedDelegateVisitor<T> : IForEachDefinedVisitor<T>
        {
            public ForEachDefinedDelegateVisitor(Action<int, T> visitor)
            {
                _visitor = visitor;
            }

            private readonly Action<int, T> _visitor;

            public void Visit(int index, T value)
            {
                _visitor(index, value);
            }
        }

#if DELEGATE_BASED_VBUFFER_UTILS
        /// <summary>
        /// Applies <paramref name="visitor"/> to every explicitly defined element of the vector,
        /// in order of index.
        /// </summary>
        public static void ForEachDefined<T>(ref VBuffer<T> a, Action<int, T> visitor)
        {
            Contracts.CheckValue(visitor, nameof(visitor));
            ForEachDefined(ref a, new ForEachDefinedDelegateVisitor<T>(visitor));
        }
#endif

        /// <summary>
        /// Applies <paramref name="visitor"/> to every explicitly defined element of the vector,
        /// in order of index. This method requires a delegate, so is somewhat slower.
        /// </summary>
        public static void ForEachDefinedSlow<T>(ref VBuffer<T> a, Action<int, T> visitor)
        {
            Contracts.CheckValue(visitor, nameof(visitor));
            ForEachDefined(ref a, new ForEachDefinedDelegateVisitor<T>(visitor));
        }

        /// <summary>
        /// Applies <paramref name="visitor"/> to every explicitly defined element of the vector,
        /// in order of index.
        /// </summary>
        public static void ForEachDefined<T, TVisitor>(ref VBuffer<T> a, TVisitor visitor) where TVisitor : struct, IForEachDefinedVisitor<T>
        {
            // Make local copies so jit can see the VBuffer fields aren't modified
            VBuffer<T> localA = a;

            T[] dataA = localA.Values;

            // REVIEW: This is analogous to an old Vector method, but is there
            // any real reason to have it given that we have the Items extension method?
            if (localA.IsDense)
            {
                if (localA.Length > dataA.Length)
                    throw new IndexOutOfRangeException();

                for (int i = 0; i < localA.Length; i++)
                {
                    visitor.Visit(i, a.Values[i]);
                }
            }
            else if (localA.Count > 0)
            {
                int[] indicesA = localA.Indices;

                if (localA.Count > indicesA.Length)
                    throw new IndexOutOfRangeException();

                for (int i = 0; i < indicesA.Length && i < localA.Count; i++)
                    visitor.Visit(indicesA[i], dataA[i]);
            }
        }

        /// <summary>
        /// Applies <paramref name="visitor"/> to every explicitly defined element of the vector,
        /// in order of index.
        /// </summary>
        public static void ForEachDefinedWithContext<T, TContext, TVisitor>(ref VBuffer<T> a, ref TContext context, TVisitor visitor) where TVisitor : struct, IForEachDefinedWithContextVisitor<T, TContext>
        {
            // Make local copies so jit can see the VBuffer fields aren't modified
            VBuffer<T> localA = a;

            T[] dataA = localA.Values;

            // REVIEW: This is analogous to an old Vector method, but is there
            // any real reason to have it given that we have the Items extension method?
            if (localA.IsDense)
            {
                if (localA.Length > dataA.Length)
                    throw new IndexOutOfRangeException();

                for (int i = 0; i < localA.Length; i++)
                {
                    visitor.Visit(i, a.Values[i], ref context);
                }
            }
            else if (localA.Count > 0)
            {
                int[] indicesA = localA.Indices;

                if (localA.Count > indicesA.Length)
                    throw new IndexOutOfRangeException();

                for (int i = 0; i < indicesA.Length && i < localA.Count; i++)
                    visitor.Visit(indicesA[i], dataA[i], ref context);
            }
        }

        public interface IPairVisitor<T>
        {
            bool Visit(int index, T value, T value2);
        }

        public interface IPairVisitor<T, TContext>
        {
            bool Visit(int index, T value, T value2, ref TContext context);
        }

        private struct PairDelegateVisitor<T> : IPairVisitor<T>
        {
            public PairDelegateVisitor(Action<int, T, T> visitor)
            {
                _visitor = visitor;
            }

            private readonly Action<int, T, T> _visitor;

            public bool Visit(int index, T value, T value2)
            {
                _visitor(index, value, value2);
                return true;
            }
        }

        private struct NonContextPairVisitor<T, TVisitor> : IPairVisitor<T, IntPtr> where TVisitor : struct, IPairVisitor<T>
        {
            public NonContextPairVisitor(TVisitor visitor)
            {
                _visitor = visitor;
            }

            private TVisitor _visitor;

            public bool Visit(int index, T value, T value2, ref IntPtr dummyvalue)
            {
                return _visitor.Visit(index, value, value2);
            }
        }

#if DELEGATE_BASED_VBUFFER_UTILS
        /// <summary>
        /// Applies the <paramref name="visitor "/>to each corresponding pair of elements
        /// where the item is emplicitly defined in the vector. By explicitly defined,
        /// we mean that for a given index <c>i</c>, both vectors have an entry in
        /// <see cref="VBuffer{T}.Values"/> corresponding to that index.
        /// </summary>
        /// <param name="a">The first vector</param>
        /// <param name="b">The second vector</param>
        /// <param name="visitor">Delegate to apply to each pair of non-zero values.
        /// This is passed the index, and two values</param>
        public static void ForEachBothDefined<T>(ref VBuffer<T> a, ref VBuffer<T> b, Action<int, T, T> visitor)
        {
            Contracts.CheckValue(visitor, nameof(visitor));
            ForEachBothDefined(ref a, ref b, new PairDelegateVisitor<T>(visitor));
        }
#endif

        /// <summary>
        /// Applies the <paramref name="visitor "/>to each corresponding pair of elements
        /// where the item is emplicitly defined in the vector. By explicitly defined,
        /// we mean that for a given index <c>i</c>, both vectors have an entry in
        /// <see cref="VBuffer{T}.Values"/> corresponding to that index.
        /// </summary>
        /// <param name="a">The first vector</param>
        /// <param name="b">The second vector</param>
        /// <param name="visitor">Operation to apply to each pair of non-zero values.
        /// This is passed the index, and two values, If the visitor returns false, shortcut and return false, otherwise return true</param>
        public static bool ForEachBothDefined<T, TVisitor>(ref VBuffer<T> a, ref VBuffer<T> b, TVisitor visitor) where TVisitor : struct, IPairVisitor<T>
        {
            IntPtr dummyValue = default(IntPtr);

            return ForEachBothDefined(ref a, ref b, ref dummyValue, new NonContextPairVisitor<T, TVisitor>(visitor));
        }

        /// <summary>
        /// Applies the <paramref name="visitor "/>to each corresponding pair of elements
        /// where the item is emplicitly defined in the vector. By explicitly defined,
        /// we mean that for a given index <c>i</c>, both vectors have an entry in
        /// <see cref="VBuffer{T}.Values"/> corresponding to that index.
        /// </summary>
        /// <param name="a">The first vector</param>
        /// <param name="b">The second vector</param>
        /// <param name="context">The context passed by reference at each visit operation</param>
        /// <param name="visitor">Operation to apply to each pair of non-zero values.
        /// This is passed the index, and two values</param>
        public static bool ForEachBothDefined<T, TContext, TVisitor>(ref VBuffer<T> a, ref VBuffer<T> b, ref TContext context, TVisitor visitor) where TVisitor : struct, IPairVisitor<T, TContext>
        {
            // Make local copies so jit can see the VBuffer fields aren't modified
            VBuffer<T> localA = a;
            VBuffer<T> localB = b;

            T[] dataA = localA.Values;
            T[] dataB = localB.Values;

            Contracts.Check(a.Length == b.Length, "Vectors must have the same dimensionality.");

            if (a.IsDense && b.IsDense)
            {
                for (int i = 0; i < a.Length; i++)
                {
                    if (!visitor.Visit(i, dataA[i], dataB[i], ref context))
                        return false;
                }
            }
            else if (b.IsDense)
            {
                int[] indicesA = a.Indices;
                for (int i = 0; i < a.Count; i++)
                {
                    if (!visitor.Visit(indicesA[i], dataA[i], dataB[indicesA[i]], ref context))
                        return false;
                }
            }
            else if (a.IsDense)
            {
                int[] indicesB = b.Indices;
                for (int i = 0; i < b.Count; i++)
                {
                    if (!visitor.Visit(indicesB[i], dataA[indicesB[i]], dataB[i], ref context))
                        return false;
                }
            }
            else
            {
                // Both sparse.
                int aI = 0;
                int bI = 0;
                while (aI < a.Count && bI < b.Count)
                {
                    int i = a.Indices[aI];
                    int j = b.Indices[bI];
                    if (i == j)
                    {
                        if (!visitor.Visit(i, dataA[aI++], dataB[bI++], ref context))
                            return false;
                    }
                    else if (i < j)
                        aI++;
                    else
                        bI++;
                }
            }

            return true;
        }

#if DELEGATE_BASED_VBUFFER_UTILS
        /// <summary>
        /// Applies the ParallelVisitor to each corresponding pair of elements where at least one is non-zero, in order of index.
        /// </summary>
        /// <param name="a">a vector</param>
        /// <param name="b">another vector</param>
        /// <param name="visitor">Function to apply to each pair of non-zero values - passed the index, and two values</param>
        public static void ForEachEitherDefined<T>(ref VBuffer<T> a, ref VBuffer<T> b, Action<int, T, T> visitor)
        {
            Contracts.CheckValue(visitor, nameof(visitor));
            ForEachEitherDefined(ref a, ref b, new PairDelegateVisitor<T>(visitor));
        }
#endif

        /// <summary>
        /// Applies the ParallelVisitor to each corresponding pair of elements where at least one is non-zero, in order of index. If any Visit operation returns false, then the function will return false otherwise it will return true.
        /// </summary>
        /// <param name="a">a vector</param>
        /// <param name="b">another vector</param>
        /// <param name="visitor">Function to apply to each pair of non-zero values - passed the index, and two values</param>
        public static bool ForEachEitherDefined<T, TVisitor>(ref VBuffer<T> a, ref VBuffer<T> b, TVisitor visitor) where TVisitor : struct, IPairVisitor<T>
        {
            IntPtr dummyValue = default(IntPtr);

            return ForEachEitherDefined(ref a, ref b, ref dummyValue, new NonContextPairVisitor<T, TVisitor>(visitor));
        }

        /// <summary>
        /// Applies the ParallelVisitor to each corresponding pair of elements where at least one is non-zero, in order of index.
        /// </summary>
        /// <param name="a">a vector</param>
        /// <param name="b">another vector</param>
        /// <param name="context">The context passed by reference at each visit operation</param>
        /// <param name="visitor">Function to apply to each pair of non-zero values - passed the index, and two values. If this function returns false, then the entire function will return false</param>
        public static bool ForEachEitherDefined<T, TContext, TVisitor>(ref VBuffer<T> a, ref VBuffer<T> b, ref TContext context, TVisitor visitor) where TVisitor : struct, IPairVisitor<T, TContext>
        {
            Contracts.Check(a.Length == b.Length, "Vectors must have the same dimensionality.");

            if (a.IsDense && b.IsDense)
            {
                for (int i = 0; i < a.Length; ++i)
                {
                    if (!visitor.Visit(i, a.Values[i], b.Values[i], ref context))
                        return false;
                }
            }
            else if (b.IsDense)
            {
                int aI = 0;
                for (int i = 0; i < b.Length; i++)
                {
                    T aVal = (aI < a.Count && i == a.Indices[aI]) ? a.Values[aI++] : default(T);
                    if (!visitor.Visit(i, aVal, b.Values[i], ref context))
                        return false;
                }
            }
            else if (a.IsDense)
            {
                int bI = 0;
                for (int i = 0; i < a.Length; i++)
                {
                    T bVal = (bI < b.Count && i == b.Indices[bI]) ? b.Values[bI++] : default(T);
                    if (!visitor.Visit(i, a.Values[i], bVal, ref context))
                        return false;
                }
            }
            else
            {
                // Both sparse
                int aI = 0;
                int bI = 0;
                while (aI < a.Count && bI < b.Count)
                {
                    int diff = a.Indices[aI] - b.Indices[bI];
                    if (diff == 0)
                    {
                        if (!visitor.Visit(b.Indices[bI], a.Values[aI], b.Values[bI], ref context))
                            return false;
                        aI++;
                        bI++;
                    }
                    else if (diff < 0)
                    {
                        if (!visitor.Visit(a.Indices[aI], a.Values[aI], default(T), ref context))
                            return false;
                        aI++;
                    }
                    else
                    {
                        if (!visitor.Visit(b.Indices[bI], default(T), b.Values[bI], ref context))
                            return false;
                        bI++;
                    }
                }

                while (aI < a.Count)
                {
                    if (!visitor.Visit(a.Indices[aI], a.Values[aI], default(T), ref context))
                        return false;
                    aI++;
                }

                while (bI < b.Count)
                {
                    if (!visitor.Visit(b.Indices[bI], default(T), b.Values[bI], ref context))
                        return false;
                    bI++;
                }
            }

            return true;
        }

        /// <summary>
        /// Sets all values in the vector to the default value for the type, without changing the
        /// density or index structure of the input array. That is to say, the count of the input
        /// vector will be the same afterwards as it was before.
        /// </summary>
        public static void Clear<T>(ref VBuffer<T> dst)
        {
            if (dst.Count == 0)
                return;
            Array.Clear(dst.Values, 0, dst.Count);
        }

        // REVIEW: Look into removing slot in this and other manipulators, so that we
        // could potentially have something around, say, skipping default entries.

        /// <summary>
        /// A delegate for functions that can change a value.
        /// </summary>
        /// <param name="slot">Index of entry</param>
        /// <param name="value">Value to change</param>
        public delegate void SlotValueManipulator<T>(int slot, ref T value);

        public interface ISlotValueManipulator<T>
        {
            void Manipulate(int slot, ref T value);
        }

        public struct SlotValueDelegateManipulator<T> : ISlotValueManipulator<T>
        {
            private SlotValueManipulator<T> _manip;

            public SlotValueDelegateManipulator(SlotValueManipulator<T> manip)
            {
                _manip = manip;
            }

            public void Manipulate(int slot, ref T value)
            {
                _manip(slot, ref value);
            }
        }

        /// <summary>
        /// A predicate on some sort of value.
        /// </summary>
        /// <param name="src">The value to test</param>
        /// <returns>The result of some sort of test from that value</returns>
        public delegate bool ValuePredicate<T>(ref T src);

        public interface IValuePredicate<T>
        {
            bool Predicate(ref T src);
        }

        public struct ValueDelegatePredicate<T> : IValuePredicate<T>
        {
            private ValuePredicate<T> _pred;

            public ValueDelegatePredicate(ValuePredicate<T> pred)
            {
                _pred = pred;
            }

            public bool Predicate(ref T value)
            {
                return _pred(ref value);
            }
        }

#if DELEGATE_BASED_VBUFFER_UTILS
        /// <summary>
        /// Applies the <paramref name="manip"/> to every explicitly defined
        /// element of the vector.
        /// </summary>
        public static void Apply<T>(ref VBuffer<T> dst, SlotValueManipulator<T> manip)
        {
            Contracts.CheckValue(manip, nameof(manip));
            Apply(ref dst, new SlotValueDelegateManipulator<T>(manip));
        }
#endif

        /// <summary>
        /// Applies the <paramref name="manip"/> to every explicitly defined
        /// element of the vector.
        /// </summary>
        public static void Apply<T, TManip>(ref VBuffer<T> dst, TManip manip) where TManip:struct, ISlotValueManipulator<T>
        {
            VBuffer<T> localDst = dst;

            T[] dataDst = localDst.Values;

            if (localDst.IsDense)
            {
                for (int i = 0; i < localDst.Length; i++)
                    manip.Manipulate(i, ref dataDst[i]);
            }
            else
            {
                int[] indicesDst = localDst.Indices;
                for (int i = 0; i < localDst.Count; i++)
                    manip.Manipulate(indicesDst[i], ref dataDst[i]);
            }
        }

        /// <summary>
        /// Applies some function on a value at a particular slot value, changing that slot value.
        /// This function will, wherever possible, not change the structure of <paramref name="dst"/>.
        /// If the vector is sparse, and the corresponding slot is not explicitly represented,
        /// then this can involve memory copying and possibly memory reallocation on <paramref name="dst"/>.
        /// However, if the item is explicitly represented, even if the item is set to the default
        /// value of <typeparamref name="T"/> it will not change the structure of <paramref name="dst"/>,
        /// in terms of sparsifying a dense array, or dropping indices.
        /// </summary>
        /// <param name="dst">The vector to modify</param>
        /// <param name="slot">The slot of the vector to modify</param>
        /// <param name="manip">The manipulation function</param>
        /// <param name="pred">A predicate that returns true if we should skip insertion of a value into
        /// sparse vector if it was default. If the predicate is null, we insert any non-default.</param>
        public static void ApplyAt<T>(ref VBuffer<T> dst, int slot, SlotValueManipulator<T> manip, ValuePredicate<T> pred = null)
        {
            Contracts.CheckParam(0 <= slot && slot < dst.Length, nameof(slot));
            Contracts.CheckValue(manip, nameof(manip));
            Contracts.CheckValueOrNull(pred);

            if (dst.IsDense)
            {
                // The vector is dense, so we can just do a direct access.
                manip(slot, ref dst.Values[slot]);
                return;
            }
            int idx = 0;
            if (dst.Count > 0 && Utils.TryFindIndexSorted(dst.Indices, 0, dst.Count, slot, out idx))
            {
                // Vector is sparse, but the item exists so we can access it.
                manip(slot, ref dst.Values[idx]);
                return;
            }
            // The vector is sparse and there is no correpsonding item, yet.
            T value = default(T);
            manip(slot, ref value);
            // If this item is not defined and it's default, no need to proceed of course.
            pred = pred ?? ((ref T val) => Comparer<T>.Default.Compare(val, default(T)) == 0);
            if (pred(ref value))
                return;
            // We have to insert this value, somehow.
            int[] indices = dst.Indices;
            T[] values = dst.Values;
            // There is a modest special case where there is exactly one free slot
            // we are modifying in the sparse vector, in which case the vector becomes
            // dense. Then there is no need to do anything with indices.
            bool needIndices = dst.Count + 1 < dst.Length;
            if (needIndices)
                Utils.EnsureSize(ref indices, dst.Count + 1, dst.Length - 1);
            Utils.EnsureSize(ref values, dst.Count + 1, dst.Length);
            if (idx != dst.Count)
            {
                // We have to do some sort of shift copy.
                if (needIndices)
                    Array.Copy(indices, idx, indices, idx + 1, dst.Count - idx);
                Array.Copy(values, idx, values, idx + 1, dst.Count - idx);
            }
            if (needIndices)
                indices[idx] = slot;
            values[idx] = value;
            dst = new VBuffer<T>(dst.Length, dst.Count + 1, values, indices);
        }

        /// <summary>
        /// Given a vector, turns it into an equivalent dense representation.
        /// </summary>
        public static void Densify<T>(ref VBuffer<T> dst)
        {
            if (dst.IsDense)
                return;
            var indices = dst.Indices;
            var values = dst.Values;
            if (Utils.Size(values) >= dst.Length)
            {
                // Densify in place.
                for (int i = dst.Count; --i >= 0; )
                {
                    Contracts.Assert(i <= indices[i]);
                    values[indices[i]] = values[i];
                }
                if (dst.Count == 0)
                    Array.Clear(values, 0, dst.Length);
                else
                {
                    int min = 0;
                    for (int ii = 0; ii < dst.Count; ++ii)
                    {
                        Array.Clear(values, min, indices[ii] - min);
                        min = indices[ii] + 1;
                    }
                    Array.Clear(values, min, dst.Length - min);
                }
            }
            else
            {
                T[] newValues = new T[dst.Length];
                for (int i = 0; i < dst.Count; ++i)
                    newValues[indices[i]] = values[i];
                values = newValues;
            }
            dst = new VBuffer<T>(dst.Length, values, indices);
        }

        /// <summary>
        /// Given a vector, ensure that the first <paramref name="denseCount"/> slots are explicitly
        /// represented.
        /// </summary>
        public static void DensifyFirst<T>(ref VBuffer<T> dst, int denseCount)
        {
            Contracts.Check(0 <= denseCount && denseCount <= dst.Length);
            if (dst.IsDense || denseCount == 0 || (dst.Count >= denseCount && dst.Indices[denseCount - 1] == denseCount - 1))
                return;
            if (denseCount == dst.Length)
            {
                Densify(ref dst);
                return;
            }

            // Densify the first BiasCount entries.
            int[] indices = dst.Indices;
            T[] values = dst.Values;
            if (indices == null)
            {
                Contracts.Assert(dst.Count == 0);
                indices = Utils.GetIdentityPermutation(denseCount);
                Utils.EnsureSize(ref values, denseCount, dst.Length, keepOld: false);
                Array.Clear(values, 0, denseCount);
                dst = new VBuffer<T>(dst.Length, denseCount, values, indices);
                return;
            }
            int lim = Utils.FindIndexSorted(indices, 0, dst.Count, denseCount);
            Contracts.Assert(lim < denseCount);
            int newLen = dst.Count + denseCount - lim;
            if (newLen == dst.Length)
            {
                Densify(ref dst);
                return;
            }
            Utils.EnsureSize(ref values, newLen, dst.Length);
            Utils.EnsureSize(ref indices, newLen, dst.Length);
            Array.Copy(values, lim, values, denseCount, dst.Count - lim);
            Array.Copy(indices, lim, indices, denseCount, dst.Count - lim);
            int i = lim - 1;
            for (int ii = denseCount; --ii >= 0; )
            {
                values[ii] = i >= 0 && indices[i] == ii ? values[i--] : default(T);
                indices[ii] = ii;
            }
            dst = new VBuffer<T>(dst.Length, newLen, values, indices);
        }

        /// <summary>
        /// Creates a maybe sparse copy of a VBuffer.
        /// Whether the created copy is sparse or not is determined by the proportion of non-default entries compared to the sparsity parameter.
        /// </summary>
        public static void CreateMaybeSparseCopy<T>(ref VBuffer<T> src, ref VBuffer<T> dst, RefPredicate<T> isDefaultPredicate, float sparsityThreshold = SparsityThreshold)
        {
            Contracts.CheckParam(0 < sparsityThreshold && sparsityThreshold < 1, nameof(sparsityThreshold));
            if (!src.IsDense || src.Length < 20)
            {
                src.CopyTo(ref dst);
                return;
            }

            int sparseCount = 0;
            var sparseCountThreshold = (int)(src.Length * sparsityThreshold);
            for (int i = 0; i < src.Length; i++)
            {
                if (!isDefaultPredicate(ref src.Values[i]))
                    sparseCount++;

                if (sparseCount > sparseCountThreshold)
                {
                    src.CopyTo(ref dst);
                    return;
                }
            }

            var indices = dst.Indices;
            var values = dst.Values;

            if (sparseCount > 0)
            {
                if (Utils.Size(values) < sparseCount)
                    values = new T[sparseCount];
                if (Utils.Size(indices) < sparseCount)
                    indices = new int[sparseCount];
                int j = 0;
                for (int i = 0; i < src.Length; i++)
                {
                    if (!isDefaultPredicate(ref src.Values[i]))
                    {
                        Contracts.Assert(j < sparseCount);
                        indices[j] = i;
                        values[j] = src.Values[i];
                        j++;
                    }
                }

                Contracts.Assert(j == sparseCount);
            }

            dst = new VBuffer<T>(src.Length, sparseCount, values, indices);
        }

        /// <summary>
        /// A delegate for functions that access an index and two corresponding
        /// values, possibly changing one of them.
        /// </summary>
        /// <param name="slot">Slot index of the entry.</param>
        /// <param name="src">Value from first vector.</param>
        /// <param name="dst">Value from second vector, which may be manipulated.</param>
        public delegate void PairManipulator<TSrc, TDst>(int slot, TSrc src, ref TDst dst);

        public interface IPairManipulator<TSrc, TDst>
        {
            void Manipulate(int slot, TSrc src, ref TDst dst);
        }

        private struct PairDelegateManipulator<TSrc, TDst> : IPairManipulator<TSrc, TDst>
        {
            private PairManipulator<TSrc, TDst> _manip;

            public PairDelegateManipulator(PairManipulator<TSrc, TDst> manip)
            {
                _manip = manip;
            }

            public void Manipulate(int slot, TSrc src, ref TDst dst)
            {
                _manip(slot, src, ref dst);
            }
        }

        /// <summary>
        /// A delegate for functions that access an index and two corresponding
        /// values, stores the result in another vector.
        /// </summary>
        /// <param name="slot">Slot index of the entry.</param>
        /// <param name="src">Value from first vector.</param>
        /// <param name="dst">Value from second vector.</param>
        /// <param name="res">The value to store the result.</param>
        public delegate void PairManipulatorCopy<TSrc, TDst>(int slot, TSrc src, TDst dst, ref TDst res);

        public interface IPairManipulatorCopy<TSrc, TDst>
        {
            void Manipulate(int slot, TSrc src, TDst dst, ref TDst res);
        }

        private struct PairDelegateManipulatorCopy<TSrc, TDst> : IPairManipulatorCopy<TSrc, TDst>
        {
            private PairManipulatorCopy<TSrc, TDst> _manip;

            public PairDelegateManipulatorCopy(PairManipulatorCopy<TSrc, TDst> manip)
            {
                _manip = manip;
            }

            public void Manipulate(int slot, TSrc src, TDst dst, ref TDst res)
            {
                _manip(slot, src, dst, ref res);
            }
        }

        private interface IBoolValue
        {
            bool Value {get;}
        }

        private struct BoolTrue : IBoolValue
        {
            public bool Value => true;
        }

        private struct BoolFalse : IBoolValue
        {
            public bool Value => false;
        }

#if DELEGATE_BASED_VBUFFER_UTILS
        /// <summary>
        /// Applies the <see cref="PairManipulator{TSrc,TDst}"/> to each pair of elements
        /// where <paramref name="src"/> is defined, in order of index. If there is
        /// some value at an index in <paramref name="dst"/> that is not defined in
        /// <paramref name="src"/>, that item remains without any further modification.
        /// If either of the vectors are dense, the resulting <paramref name="dst"/>
        /// will be dense. Otherwise, if both are sparse, the output will be sparse iff
        /// there is any slot that is not explicitly represented in either vector.
        /// </summary>
        /// <param name="src">Argument vector, whose elements are only read</param>
        /// <param name="dst">Argument vector, that could change</param>
        /// <param name="manip">Function to apply to each pair of elements</param>
        public static void ApplyWith<TSrc, TDst>(ref VBuffer<TSrc> src, ref VBuffer<TDst> dst, PairManipulator<TSrc, TDst> manip)
        {
            Contracts.CheckValue(manip, nameof(manip));
            ApplyWithCore(ref src, ref dst, new PairDelegateManipulator<TSrc, TDst>(manip), outer: new BoolFalse());
        }
#endif

        /// <summary>
        /// Applies the <see cref="PairManipulator{TSrc,TDst}"/> to each pair of elements
        /// where <paramref name="src"/> is defined, in order of index. If there is
        /// some value at an index in <paramref name="dst"/> that is not defined in
        /// <paramref name="src"/>, that item remains without any further modification.
        /// If either of the vectors are dense, the resulting <paramref name="dst"/>
        /// will be dense. Otherwise, if both are sparse, the output will be sparse iff
        /// there is any slot that is not explicitly represented in either vector.
        /// </summary>
        /// <param name="src">Argument vector, whose elements are only read</param>
        /// <param name="dst">Argument vector, that could change</param>
        /// <param name="manip">Function to apply to each pair of elements</param>
        public static void ApplyWith<TSrc, TDst, TPairManipulator>(ref VBuffer<TSrc> src, ref VBuffer<TDst> dst, TPairManipulator manip)
            where TPairManipulator : struct, IPairManipulator<TSrc, TDst>
        {
            ApplyWithCore(ref src, ref dst, manip, outer: new BoolFalse());
        }

#if DELEGATE_BASED_VBUFFER_UTILS
        /// <summary>
        /// Applies the <see cref="PairManipulator{TSrc,TDst}"/> to each pair of elements
        /// where <paramref name="src"/> is defined, in order of index. It stores the result
        /// in another vector. If there is some value at an index in <paramref name="dst"/>
        /// that is not defined in <paramref name="src"/>, that slot value is copied to the
        /// corresponding slot in the result vector without any further modification.
        /// If either of the vectors are dense, the resulting <paramref name="res"/>
        /// will be dense. Otherwise, if both are sparse, the output will be sparse iff
        /// there is any slot that is not explicitly represented in either vector.
        /// </summary>
        /// <param name="src">Argument vector, whose elements are only read</param>
        /// <param name="dst">Argument vector, whose elements are only read</param>
        /// <param name="res">Result vector</param>
        /// <param name="manip">Function to apply to each pair of elements</param>
        public static void ApplyWithCopy<TSrc, TDst>(ref VBuffer<TSrc> src, ref VBuffer<TDst> dst, ref VBuffer<TDst> res, PairManipulatorCopy<TSrc, TDst> manip)
        {
            Contracts.CheckValue(manip, nameof(manip));
            ApplyWithCoreCopy(ref src, ref dst, ref res, new PairDelegateManipulatorCopy<TSrc, TDst>(manip), outer: new BoolFalse());
        }
#endif

        /// <summary>
        /// Applies the <see cref="PairManipulator{TSrc,TDst}"/> to each pair of elements
        /// where <paramref name="src"/> is defined, in order of index. It stores the result
        /// in another vector. If there is some value at an index in <paramref name="dst"/>
        /// that is not defined in <paramref name="src"/>, that slot value is copied to the
        /// corresponding slot in the result vector without any further modification.
        /// If either of the vectors are dense, the resulting <paramref name="res"/>
        /// will be dense. Otherwise, if both are sparse, the output will be sparse iff
        /// there is any slot that is not explicitly represented in either vector.
        /// </summary>
        /// <param name="src">Argument vector, whose elements are only read</param>
        /// <param name="dst">Argument vector, whose elements are only read</param>
        /// <param name="res">Result vector</param>
        /// <param name="manip">Function to apply to each pair of elements</param>
        public static void ApplyWithCopy<TSrc, TDst, TPairManipulator>(ref VBuffer<TSrc> src, ref VBuffer<TDst> dst, ref VBuffer<TDst> res, TPairManipulator manip)
            where TPairManipulator : struct, IPairManipulatorCopy<TSrc, TDst>
        {
            ApplyWithCoreCopy(ref src, ref dst, ref res, manip, outer: new BoolFalse());
        }

#if DELEGATE_BASED_VBUFFER_UTILS
        /// <summary>
        /// Applies the <see cref="PairManipulator{TSrc,TDst}"/> to each pair of elements
        /// where either <paramref name="src"/> or <paramref name="dst"/>, has an element
        /// defined at that index. If either of the vectors are dense, the resulting
        /// <paramref name="dst"/> will be dense. Otherwise, if both are sparse, the output
        /// will be sparse iff there is any slot that is not explicitly represented in
        /// either vector.
        /// </summary>
        /// <param name="src">Argument vector, whose elements are only read</param>
        /// <param name="dst">Argument vector, that could change</param>
        /// <param name="manip">Function to apply to each pair of elements</param>
        public static void ApplyWithEitherDefined<TSrc, TDst>(ref VBuffer<TSrc> src, ref VBuffer<TDst> dst, PairManipulator<TSrc, TDst> manip)
        {
            Contracts.CheckValue(manip, nameof(manip));
            ApplyWithCore(ref src, ref dst, new PairDelegateManipulator<TSrc, TDst>(manip), outer: new BoolTrue());
        }
#endif

        /// <summary>
        /// Applies the <see cref="PairManipulator{TSrc,TDst}"/> to each pair of elements
        /// where either <paramref name="src"/> or <paramref name="dst"/>, has an element
        /// defined at that index. If either of the vectors are dense, the resulting
        /// <paramref name="dst"/> will be dense. Otherwise, if both are sparse, the output
        /// will be sparse iff there is any slot that is not explicitly represented in
        /// either vector.
        /// </summary>
        /// <param name="src">Argument vector, whose elements are only read</param>
        /// <param name="dst">Argument vector, that could change</param>
        /// <param name="manip">Function to apply to each pair of elements</param>
        public static void ApplyWithEitherDefined<TSrc, TDst, TPairManipulator>(ref VBuffer<TSrc> src, ref VBuffer<TDst> dst, TPairManipulator manip)
            where TPairManipulator : struct, IPairManipulator<TSrc, TDst>
        {
            ApplyWithCore(ref src, ref dst, manip, outer: new BoolTrue());
        }

#if DELEGATE_BASED_VBUFFER_UTILS
        /// <summary>
        /// Applies the <see cref="PairManipulator{TSrc,TDst}"/> to each pair of elements
        /// where either <paramref name="src"/> or <paramref name="dst"/>, has an element
        /// defined at that index. It stores the result in another vector <paramref name="res"/>.
        /// If either of the vectors are dense, the resulting <paramref name="res"/>
        /// will be dense. Otherwise, if both are sparse, the output will be sparse iff
        /// there is any slot that is not explicitly represented in either vector.
        /// </summary>
        /// <param name="src">Argument vector, whose elements are only read</param>
        /// <param name="dst">Argument vector, whose elements are only read</param>
        /// <param name="res">Result vector</param>
        /// <param name="manip">Function to apply to each pair of elements</param>
        public static void ApplyWithEitherDefinedCopy<TSrc, TDst>(ref VBuffer<TSrc> src, ref VBuffer<TDst> dst, ref VBuffer<TDst> res, PairManipulatorCopy<TSrc, TDst> manip)
        {
            Contracts.CheckValue(manip, nameof(manip));
            ApplyWithCoreCopy(ref src, ref dst, ref res, new PairDelegateManipulatorCopy<TSrc, TDst>(manip), outer: new BoolTrue());
        }
#endif

        /// <summary>
        /// Applies the <see cref="PairManipulator{TSrc,TDst}"/> to each pair of elements
        /// where either <paramref name="src"/> or <paramref name="dst"/>, has an element
        /// defined at that index. It stores the result in another vector <paramref name="res"/>.
        /// If either of the vectors are dense, the resulting <paramref name="res"/>
        /// will be dense. Otherwise, if both are sparse, the output will be sparse iff
        /// there is any slot that is not explicitly represented in either vector.
        /// </summary>
        /// <param name="src">Argument vector, whose elements are only read</param>
        /// <param name="dst">Argument vector, whose elements are only read</param>
        /// <param name="res">Result vector</param>
        /// <param name="manip">Function to apply to each pair of elements</param>
        public static void ApplyWithEitherDefinedCopy<TSrc, TDst, TPairManipulator>(ref VBuffer<TSrc> src, ref VBuffer<TDst> dst, ref VBuffer<TDst> res, TPairManipulator manip)
            where TPairManipulator : struct, IPairManipulatorCopy<TSrc, TDst>
        {
            ApplyWithCoreCopy(ref src, ref dst, ref res, manip, outer: new BoolTrue());
        }

        /// <summary>
        /// The actual implementation of
        /// <see cref="VBufferUtils.ApplyWith{TSrc, TDst, TPairManipulator}"/>, and
        /// <see cref="ApplyWithEitherDefined{TSrc,TDst, TPairManipulator}"/>, that has
        /// internal branches on the implementation where necessary depending on whether
        /// this is an inner or outer join of the indices of <paramref name="src"/> on
        /// <paramref name="dst"/>.
        /// </summary>
        private static void ApplyWithCore<TSrc, TDst, TPairManipulator, TBoolValue>(ref VBuffer<TSrc> src, ref VBuffer<TDst> dst, TPairManipulator manip, TBoolValue outer)
            where TPairManipulator : struct, IPairManipulator<TSrc, TDst>
            where TBoolValue : struct, IBoolValue
        {
            Contracts.Check(src.Length == dst.Length, "Vectors must have the same dimensionality.");

            // We handle all of the permutations of the density/sparsity of src/dst through
            // special casing below. Each subcase in turn handles appropriately the treatment
            // of the "outer" parameter. There are nine, top level cases. Each case is
            // considered in this order.

            // 1. src.Count == 0.
            // 2. src.Dense.
            // 3. dst.Dense.
            // 4. dst.Count == 0.

            // Beyond this point the cases can assume both src/dst are sparse non-empty vectors.
            // We then calculate the size of the resulting output array, then use that to fall
            // through to more special cases.

            // 5. The union will result in dst becoming dense. So just densify it, then recurse.
            // 6. Neither src nor dst's indices is a subset of the other.
            // 7. The two sets of indices are identical.
            // 8. src's indices are a subset of dst's.
            // 9. dst's indices are a subset of src's.

            // Each one of these subcases also separately handles the "outer" parameter, if
            // necessary. It is unnecessary if src's indices form a superset (proper or improper)
            // of dst's indices. So, for example, cases 2, 4, 7, 9 do not require special handling.
            // Case 5 does not require special handling, because it falls through to other cases
            // that do the special handling for them.

            VBuffer<TSrc> localSrc = src;
            VBuffer<TDst> localDst = dst;
            if (localSrc.Count == 0)
            {
                // Major case 1, with src.Count == 0.
                if (!outer.Value)
                    return;
                if (localDst.IsDense)
                {
                    for (int i = 0; i < localDst.Length; i++)
                        manip.Manipulate(i, default(TSrc), ref localDst.Values[i]);
                }
                else
                {
                    for (int i = 0; i < localDst.Count; i++)
                        manip.Manipulate(localDst.Indices[i], default(TSrc), ref localDst.Values[i]);
                }
                return;
            }

            if (localSrc.IsDense)
            {
                // Major case 2, with src.Dense.
                if (!localDst.IsDense)
                    Densify(ref localDst);
                // Both are now dense. Both cases of outer are covered.
                for (int i = 0; i < localSrc.Length; i++)
                    manip.Manipulate(i, localSrc.Values[i], ref localDst.Values[i]);
                return;
            }

            if (localDst.IsDense)
            {
                // Major case 3, with dst.Dense. Note that !a.Dense.
                if (outer.Value)
                {
                    int sI = 0;
                    int sIndex = localSrc.Indices[sI];
                    for (int i = 0; i < localDst.Length; ++i)
                    {
                        if (i == sIndex)
                        {
                            manip.Manipulate(i, localSrc.Values[sI], ref localDst.Values[i]);
                            sIndex = ++sI == localSrc.Count ? localSrc.Length : localSrc.Indices[sI];
                        }
                        else
                            manip.Manipulate(i, default(TSrc), ref localDst.Values[i]);
                    }
                }
                else
                {
                    for (int i = 0; i < localSrc.Count; i++)
                        manip.Manipulate(localSrc.Indices[i], localSrc.Values[i], ref localDst.Values[localSrc.Indices[i]]);
                }
                return;
            }

            if (localDst.Count == 0)
            {
                // Major case 4, with dst empty. Note that !src.Dense.
                // Neither is dense, and dst is empty. Both cases of outer are covered.
                var values = localDst.Values;
                var indices = localDst.Indices;
                values = Utils.EnsureSize(values, localSrc.Count, localSrc.Length);
                Array.Clear(values, 0, localSrc.Count);
                indices = Utils.EnsureSize(indices, localSrc.Count, localSrc.Length);
                for (int i = 0; i < localSrc.Count; i++)
                    manip.Manipulate(indices[i] = localSrc.Indices[i], localSrc.Values[i], ref values[i]);
                localDst = new VBuffer<TDst>(localSrc.Length, localSrc.Count, values, indices);
                return;
            }

            // Beyond this point, we can assume both a and b are sparse with positive count.
            int dI = 0;
            int newCount = localDst.Count;
            // Try to find each src index in dst indices, counting how many more we'll add.
            for (int sI = 0; sI < localSrc.Count; sI++)
            {
                int sIndex = localSrc.Indices[sI];
                while (dI < localDst.Count && localDst.Indices[dI] < sIndex)
                    dI++;
                if (dI == localDst.Count)
                {
                    newCount += localSrc.Count - sI;
                    break;
                }
                if (localDst.Indices[dI] == sIndex)
                    dI++;
                else
                    newCount++;
            }
            Contracts.Assert(newCount > 0);
            Contracts.Assert(0 < localSrc.Count && localSrc.Count <= newCount);
            Contracts.Assert(0 < localDst.Count && localDst.Count <= newCount);

            // REVIEW: Densify above a certain threshold, not just if
            // the output will necessarily become dense? But then we get into
            // the dubious business of trying to pick the "right" densification
            // threshold.
            if (newCount == localDst.Length)
            {
                // Major case 5, dst will become dense through the application of
                // this. Just recurse one level so one of the initial conditions
                // can catch it, specifically, the major case 3.

                // This is unnecessary -- falling through to the sparse code will
                // actually handle this case just fine -- but it is more efficient.
                Densify(ref dst);
                ApplyWithCore(ref src, ref dst, manip, outer);
                return;
            }

            if (newCount != localSrc.Count && newCount != localDst.Count)
            {
                // Major case 6, neither set of indices is a subset of the other.
                // This subcase used to fall through to another subcase, but this
                // proved to be inefficient so we go to the little bit of extra work
                // to handle it here.

                var indices = localDst.Indices;
                var values = localDst.Values;
                indices = Utils.EnsureSize(indices, newCount, localDst.Length, keepOld: false);
                values = Utils.EnsureSize(values, newCount, localDst.Length, keepOld: false);
                int sI = localSrc.Count - 1;
                dI = localDst.Count - 1;
                int sIndex = localSrc.Indices[sI];
                int dIndex = localDst.Indices[dI];

                // Go from the end, so that even if we're writing over dst's vectors in
                // place, we do not corrupt the data as we are reorganizing it.
                for (int i = newCount; --i >= 0; )
                {
                    if (sIndex < dIndex)
                    {
                        indices[i] = dIndex;
                        values[i] = localDst.Values[dI];
                        if (outer.Value)
                            manip.Manipulate(dIndex, default(TSrc), ref values[i]);
                        dIndex = --dI >= 0 ? localDst.Indices[dI] : -1;
                    }
                    else if (sIndex > dIndex)
                    {
                        indices[i] = sIndex;
                        values[i] = default(TDst);
                        manip.Manipulate(sIndex, localSrc.Values[sI], ref values[i]);
                        sIndex = --sI >= 0 ? localSrc.Indices[sI] : -1;
                    }
                    else
                    {
                        // We should not have run past the beginning, due to invariants.
                        Contracts.Assert(sIndex >= 0);
                        Contracts.Assert(sIndex == dIndex);
                        indices[i] = dIndex;
                        values[i] = localDst.Values[dI];
                        manip.Manipulate(sIndex, localSrc.Values[sI], ref values[i]);
                        sIndex = --sI >= 0 ? localSrc.Indices[sI] : -1;
                        dIndex = --dI >= 0 ? localDst.Indices[dI] : -1;
                    }
                }
                localDst = new VBuffer<TDst>(localDst.Length, newCount, values, indices);
                return;
            }

            if (newCount == localDst.Count)
            {
                if (newCount == localSrc.Count)
                {
                    // Major case 7, the set of indices is the same for src and dst.
                    Contracts.Assert(localSrc.Count == localDst.Count);
                    for (int i = 0; i < localSrc.Count; i++)
                    {
                        Contracts.Assert(localSrc.Indices[i] == localDst.Indices[i]);
                        manip.Manipulate(localSrc.Indices[i], localSrc.Values[i], ref localDst.Values[i]);
                    }
                    return;
                }
                // Major case 8, the indices of src must be a subset of dst's indices.
                Contracts.Assert(newCount > localSrc.Count);
                dI = 0;
                if (outer.Value)
                {
                    int sI = 0;
                    int sIndex = localSrc.Indices[sI];
                    for (int i = 0; i < localDst.Count; ++i)
                    {
                        if (localDst.Indices[i] == sIndex)
                        {
                            manip.Manipulate(sIndex, localSrc.Values[sI], ref localDst.Values[i]);
                            sIndex = ++sI == localSrc.Count ? localSrc.Length : localSrc.Indices[sI];
                        }
                        else
                            manip.Manipulate(localDst.Indices[i], default(TSrc), ref localDst.Values[i]);
                    }
                }
                else
                {
                    for (int sI = 0; sI < localSrc.Count; sI++)
                    {
                        int sIndex = localSrc.Indices[sI];
                        while (localDst.Indices[dI] < sIndex)
                            dI++;
                        Contracts.Assert(localDst.Indices[dI] == sIndex);
                        manip.Manipulate(sIndex, localSrc.Values[sI], ref localDst.Values[dI++]);
                    }
                }
                return;
            }

            if (newCount == localSrc.Count)
            {
                // Major case 9, the indices of dst must be a subset of src's indices. Both cases of outer are covered.

                // First do a "quasi" densification of dst, by making the indices
                // of dst correspond to those in src.
                int sI = 0;
                for (dI = 0; dI < localDst.Count; ++dI)
                {
                    int bIndex = localDst.Indices[dI];
                    while (localSrc.Indices[sI] < bIndex)
                        sI++;
                    Contracts.Assert(localSrc.Indices[sI] == bIndex);
                    localDst.Indices[dI] = sI++;
                }
                localDst = new VBuffer<TDst>(newCount, localDst.Count, localDst.Values, localDst.Indices);
                Densify(ref localDst);
                int[] indices = localDst.Indices;
                indices = Utils.EnsureSize(indices, localSrc.Count, localSrc.Length, keepOld: false);
                Array.Copy(localSrc.Indices, indices, newCount);
                localDst = new VBuffer<TDst>(localSrc.Length, newCount, localDst.Values, indices);
                for (sI = 0; sI < localSrc.Count; sI++)
                    manip.Manipulate(localSrc.Indices[sI], localSrc.Values[sI], ref localDst.Values[sI]);
                return;
            }

            Contracts.Assert(false);
        }

        /// <summary>
        /// The actual implementation of
        /// <see cref="VBufferUtils.ApplyWithCopy{TSrc, TDst, TPairManipulator}"/>, and
        /// <see cref="ApplyWithEitherDefinedCopy{TSrc,TDst, TPairManipulator}"/>, that has internal
        /// branches on the implementation where necessary depending on whether this is an inner or outer join of the
        /// indices of <paramref name="src"/> on <paramref name="dst"/>.
        /// </summary>
        private static void ApplyWithCoreCopy<TSrc, TDst, TPairManipulator, TBoolValue>(ref VBuffer<TSrc> src, ref VBuffer<TDst> dst, ref VBuffer<TDst> res, TPairManipulator manip, TBoolValue outer)
            where TPairManipulator : struct, IPairManipulatorCopy<TSrc, TDst>
            where TBoolValue : struct, IBoolValue
        {
            Contracts.Check(src.Length == dst.Length, "Vectors must have the same dimensionality.");
            Contracts.Assert(Utils.Size(src.Values) >= src.Count);
            Contracts.Assert(Utils.Size(dst.Values) >= dst.Count);
            int length = src.Length;

            if (dst.Count == 0)
            {
                if (src.Count == 0)
                    res = new VBuffer<TDst>(length, 0, res.Values, res.Indices);
                else if (src.IsDense)
                {
                    Contracts.Assert(src.Count == src.Length);
                    TDst[] resValues = Utils.Size(res.Values) >= length ? res.Values : new TDst[length];
                    for (int i = 0; i < length; i++)
                        manip.Manipulate(i, src.Values[i], default(TDst), ref resValues[i]);
                    res = new VBuffer<TDst>(length, resValues, res.Indices);
                }
                else
                {
                    // src is non-empty sparse.
                    int count = src.Count;
                    Contracts.Assert(0 < count && count < length);
                    int[] resIndices = Utils.Size(res.Indices) >= count ? res.Indices : new int[count];
                    TDst[] resValues = Utils.Size(res.Values) >= count ? res.Values : new TDst[count];
                    Array.Copy(src.Indices, resIndices, count);
                    for (int ii = 0; ii < count; ii++)
                    {
                        int i = src.Indices[ii];
                        resIndices[ii] = i;
                        manip.Manipulate(i, src.Values[ii], default(TDst), ref resValues[ii]);
                    }
                    res = new VBuffer<TDst>(length, count, resValues, resIndices);
                }
            }
            else if (dst.IsDense)
            {
                TDst[] resValues = Utils.Size(res.Values) >= length ? res.Values : new TDst[length];
                if (src.Count == 0)
                {
                    if (outer.Value)
                    {
                        // Apply manip to all slots, as all slots of dst are defined.
                        for (int j = 0; j < length; j++)
                            manip.Manipulate(j, default(TSrc), dst.Values[j], ref resValues[j]);
                    }
                    else
                    {
                        // Copy only. No slot of src is defined.
                        for (int j = 0; j < length; j++)
                            resValues[j] = dst.Values[j];
                    }
                    res = new VBuffer<TDst>(length, resValues, res.Indices);
                }
                else if (src.IsDense)
                {
                    Contracts.Assert(src.Count == src.Length);
                    for (int i = 0; i < length; i++)
                        manip.Manipulate(i, src.Values[i], dst.Values[i], ref resValues[i]);
                    res = new VBuffer<TDst>(length, resValues, res.Indices);
                }
                else
                {
                    // src is sparse and non-empty.
                    int count = src.Count;
                    Contracts.Assert(0 < count && count < length);

                    int ii = 0;
                    int i = src.Indices[ii];
                    if (outer.Value)
                    {
                        // All slots of dst are defined. Always apply manip.
                        for (int j = 0; j < length; j++)
                        {
                            if (j == i)
                            {
                                manip.Manipulate(j, src.Values[ii], dst.Values[j], ref resValues[j]);
                                i = ++ii == count ? length : src.Indices[ii];
                            }
                            else
                                manip.Manipulate(j, default(TSrc), dst.Values[j], ref resValues[j]);
                        }
                    }
                    else
                    {
                        // Only apply manip for those slots where src is defined. Otherwise just copy.
                        for (int j = 0; j < length; j++)
                        {
                            if (j == i)
                            {
                                manip.Manipulate(j, src.Values[ii], dst.Values[j], ref resValues[j]);
                                i = ++ii == count ? length : src.Indices[ii];
                            }
                            else
                                resValues[j] = dst.Values[j];
                        }
                    }
                    res = new VBuffer<TDst>(length, resValues, res.Indices);
                }
            }
            else
            {
                // dst is non-empty sparse
                int dstCount = dst.Count;
                Contracts.Assert(dstCount > 0);
                if (src.Count == 0)
                {
                    int[] resIndices = Utils.Size(res.Indices) >= dstCount ? res.Indices : new int[dstCount];
                    TDst[] resValues = Utils.Size(res.Values) >= dstCount ? res.Values : new TDst[dstCount];
                    if (outer.Value)
                    {
                        for (int jj = 0; jj < dstCount; jj++)
                        {
                            int j = dst.Indices[jj];
                            resIndices[jj] = j;
                            manip.Manipulate(j, default(TSrc), dst.Values[jj], ref resValues[jj]);
                        }
                    }
                    else
                    {
                        for (int jj = 0; jj < dstCount; jj++)
                        {
                            resIndices[jj] = dst.Indices[jj];
                            resValues[jj] = dst.Values[jj];
                        }
                    }
                    res = new VBuffer<TDst>(length, dstCount, resValues, resIndices);
                }
                else if (src.IsDense)
                {
                    // res will be dense.
                    TDst[] resValues = Utils.Size(res.Values) >= length ? res.Values : new TDst[length];
                    int jj = 0;
                    int j = dst.Indices[jj];
                    for (int i = 0; i < length; i++)
                    {
                        if (i == j)
                        {
                            manip.Manipulate(i, src.Values[i], dst.Values[jj], ref resValues[i]);
                            j = ++jj == dstCount ? length : dst.Indices[jj];
                        }
                        else
                            manip.Manipulate(i, src.Values[i], default(TDst), ref resValues[i]);
                    }
                    res = new VBuffer<TDst>(length, resValues, res.Indices);
                }
                else
                {
                    // Both src and dst are non-empty sparse.
                    Contracts.Assert(src.Count > 0);

                    // Find the count of result, which is the size of the union of the indices set of src and dst.
                    int resCount = dstCount;
                    for (int ii = 0, jj = 0; ii < src.Count; ii++)
                    {
                        int i = src.Indices[ii];
                        while (jj < dst.Count && dst.Indices[jj] < i)
                            jj++;
                        if (jj == dst.Count)
                        {
                            resCount += src.Count - ii;
                            break;
                        }
                        if (dst.Indices[jj] == i)
                            jj++;
                        else
                            resCount++;
                    }

                    Contracts.Assert(0 < resCount && resCount <= length);
                    Contracts.Assert(resCount <= src.Count + dstCount);
                    Contracts.Assert(src.Count <= resCount);
                    Contracts.Assert(dstCount <= resCount);

                    if (resCount == length)
                    {
                        // result will become dense.
                        // This is unnecessary -- falling through to the sparse code will
                        // actually handle this case just fine -- but it is more efficient.
                        Densify(ref dst);
                        ApplyWithCoreCopy(ref src, ref dst, ref res, manip, outer);
                    }
                    else
                    {
                        int[] resIndices = Utils.Size(res.Indices) >= resCount ? res.Indices : new int[resCount];
                        TDst[] resValues = Utils.Size(res.Values) >= resCount ? res.Values : new TDst[resCount];

                        int ii = 0;
                        int i = src.Indices[ii];
                        int jj = 0;
                        int j = dst.Indices[jj];

                        for (int kk = 0; kk < resCount; kk++)
                        {
                            Contracts.Assert(i < length || j < length);
                            if (i == j)
                            {
                                // Slot (i == j) both defined in src and dst. Apply manip.
                                resIndices[kk] = i;
                                manip.Manipulate(i, src.Values[ii], dst.Values[jj], ref resValues[kk]);
                                i = ++ii == src.Count ? length : src.Indices[ii];
                                j = ++jj == dstCount ? length : dst.Indices[jj];
                            }
                            else if (i < j)
                            {
                                // Slot i defined only in src, but not in dst. Apply manip.
                                resIndices[kk] = i;
                                manip.Manipulate(i, src.Values[ii], default(TDst), ref resValues[kk]);
                                i = ++ii == src.Count ? length : src.Indices[ii];
                            }
                            else
                            {
                                // Slot j defined only in dst, but not in src. Apply manip if outer.
                                // Otherwise just copy.
                                resIndices[kk] = j;
                                // REVIEW: Should we move checking of outer outside the loop?
                                if (outer.Value)
                                    manip.Manipulate(j, default(TSrc), dst.Values[jj], ref resValues[kk]);
                                else
                                    resValues[kk] = dst.Values[jj];
                                j = ++jj == dstCount ? length : dst.Indices[jj];
                            }
                        }

                        Contracts.Assert(ii == src.Count && jj == dstCount);
                        Contracts.Assert(i == length && j == length);
                        res = new VBuffer<TDst>(length, resCount, resValues, resIndices);
                    }
                }
            }
        }

        public interface IDstProducingVisitor<TSrc, TDst>
        {
            TDst Visit(int index, TSrc value);
        }

        public interface IDstProducingVisitor<TSrc, TDst, TContext>
        {
            TDst Visit(int index, TSrc value, ref TContext context);
        }

        private struct DstProducingDelegateVisitor<TSrc, TDst> : IDstProducingVisitor<TSrc, TDst>
        {
            public DstProducingDelegateVisitor(Func<int, TSrc, TDst> visitor)
            {
                _visitor = visitor;
            }

            private readonly Func<int, TSrc, TDst> _visitor;

            public TDst Visit(int index, TSrc value)
            {
                return _visitor(index, value);
            }
        }

        private struct NonContextDstProducingVisitor<TSrc, TDst, TVisitor> : IDstProducingVisitor<TSrc, TDst, IntPtr> where TVisitor : struct, IDstProducingVisitor<TSrc, TDst>
        {
            public NonContextDstProducingVisitor(TVisitor visitor)
            {
                _visitor = visitor;
            }

            private TVisitor _visitor;

            public TDst Visit(int index, TSrc value, ref IntPtr dummyvalue)
            {
                return _visitor.Visit(index, value);
            }
        }

#if DELEGATE_BASED_VBUFFER_UTILS
        /// <summary>
        /// Applies a function to explicitly defined elements in a vector <paramref name="src"/>,
        /// storing the result in <paramref name="dst"/>, overwriting any of its existing contents.
        /// The contents of <paramref name="dst"/> do not affect calculation. If you instead wish
        /// to calculate a function that reads and writes <paramref name="dst"/>, see
        /// <see cref="ApplyWith{TSrc,TDst,TPairManipulator}"/> and <see cref="ApplyWithEitherDefined{TSrc,TDst,TPairManipulator}"/>. Post-operation,
        /// <paramref name="dst"/> will be dense iff <paramref name="src"/> is dense.
        /// </summary>
        /// <seealso cref="ApplyWith{TSrc,TDst,TPairManipulator}"/>
        /// <seealso cref="ApplyWithEitherDefined{TSrc,TDst,TPairManipulator}"/>
        public static void ApplyIntoEitherDefined<TSrc, TDst>(ref VBuffer<TSrc> src, ref VBuffer<TDst> dst, Func<int, TSrc, TDst> func)
        {
            Contracts.CheckValue(func, nameof(func));
            ApplyIntoEitherDefined(ref src, ref dst, new DstProducingDelegateVisitor<TSrc, TDst>(func));
        }
#endif

        public static void ApplyIntoEitherDefined<TSrc, TDst, TVisitor>(ref VBuffer<TSrc> src, ref VBuffer<TDst> dst, TVisitor visitor) where TVisitor : struct, IDstProducingVisitor<TSrc, TDst>
        {
            IntPtr dummyValue = default(IntPtr);

            ApplyIntoEitherDefined(ref src, ref dst, ref dummyValue, new NonContextDstProducingVisitor<TSrc, TDst, TVisitor>(visitor));
        }

        public static void ApplyIntoEitherDefined<TSrc, TDst, TContext, TVisitor>(ref VBuffer<TSrc> src, ref VBuffer<TDst> dst, ref TContext context, TVisitor visitor) where TVisitor : struct, IDstProducingVisitor<TSrc, TDst, TContext>
        {
            VBuffer<TSrc> localSrc = src;
            int[] indices = dst.Indices;

            // REVIEW: The analogous WritableVector method insisted on
            // equal lengths, but I don't care here.
            if (localSrc.Count == 0)
            {
                dst = new VBuffer<TDst>(localSrc.Length, localSrc.Count, dst.Values, indices);
                return;
            }

            TDst[] values = Utils.EnsureSize(dst.Values, localSrc.Count, localSrc.Length, keepOld: false);
            TSrc[] valuesSrc = localSrc.Values;
            if (localSrc.IsDense)
            {
                for (int i = 0; i < localSrc.Length; ++i)
                    values[i] = visitor.Visit(i, valuesSrc[i], ref context);
            }
            else
            {
                indices = Utils.EnsureSize(indices, localSrc.Count, localSrc.Length, keepOld: false);
                Array.Copy(localSrc.Indices, indices, localSrc.Count);
                for (int i = 0; i < localSrc.Count; ++i)
                    values[i] = visitor.Visit(indices[i], valuesSrc[i], ref context);
            }
            dst = new VBuffer<TDst>(localSrc.Length, localSrc.Count, values, indices);
        }

        public interface IDstProducingPairVisitor<TSrc1, TSrc2, TDst>
        {
            TDst Visit(int index, TSrc1 value, TSrc2 value2);
        }

        public interface IDstProducingPairVisitor<TSrc1, TSrc2, TDst, TContext>
        {
            TDst Visit(int index, TSrc1 value, TSrc2 value2, ref TContext context);
        }

        private struct DstProducingPairDelegateVisitor<TSrc1, TSrc2, TDst> : IDstProducingPairVisitor<TSrc1, TSrc2, TDst>
        {
            public DstProducingPairDelegateVisitor(Func<int, TSrc1, TSrc2, TDst> visitor)
            {
                _visitor = visitor;
            }

            private readonly Func<int, TSrc1, TSrc2, TDst> _visitor;

            public TDst Visit(int index, TSrc1 value, TSrc2 value2)
            {
                return _visitor(index, value, value2);
            }
        }

        private struct NonContextDstProducingPairVisitor<TSrc1, TSrc2, TDst, TVisitor> : IDstProducingPairVisitor<TSrc1, TSrc2, TDst, IntPtr> where TVisitor : struct, IDstProducingPairVisitor<TSrc1, TSrc2, TDst>
        {
            public NonContextDstProducingPairVisitor(TVisitor visitor)
            {
                _visitor = visitor;
            }

            private TVisitor _visitor;

            public TDst Visit(int index, TSrc1 value, TSrc2 value2, ref IntPtr dummyvalue)
            {
                return _visitor.Visit(index, value, value2);
            }
        }

#if DELEGATE_BASED_VBUFFER_UTILS
        /// <summary>
        /// Applies a function <paramref name="func"/> to two vectors, storing the result in
        /// <paramref name="dst"/>, whose existing contents are discarded and overwritten. The
        /// function is called for every index value that appears in either <paramref name="a"/>
        /// or <paramref name="b"/>. If either of the two inputs is dense, the output will
        /// necessarily be dense. Otherwise, if both are sparse, the output will be sparse iff
        /// there is any slot that is not explicitly represented in either vector.
        /// </summary>
        public static void ApplyInto<TSrc1, TSrc2, TDst>(ref VBuffer<TSrc1> a, ref VBuffer<TSrc2> b, ref VBuffer<TDst> dst, Func<int, TSrc1, TSrc2, TDst> func)
        {
            Contracts.CheckValue(func, nameof(func));
            ApplyInto(ref a, ref b, ref dst, new DstProducingPairDelegateVisitor<TSrc1, TSrc2, TDst>(func));
        }
#endif

        public static void ApplyInto<TSrc1, TSrc2, TDst, TVisitor>(ref VBuffer<TSrc1> a, ref VBuffer<TSrc2> b, ref VBuffer<TDst> dst, TVisitor visitor) where TVisitor : struct, IDstProducingPairVisitor<TSrc1, TSrc2, TDst>
        {
            IntPtr dummyValue = default(IntPtr);

            ApplyInto(ref a, ref b, ref dst, ref dummyValue, new NonContextDstProducingPairVisitor<TSrc1, TSrc2, TDst, TVisitor>(visitor));
        }

        public static void ApplyInto<TSrc1, TSrc2, TDst, TContext, TVisitor>(ref VBuffer<TSrc1> aInput, ref VBuffer<TSrc2> bInput, ref VBuffer<TDst> dst, ref TContext context, TVisitor visitor) where TVisitor : struct, IDstProducingPairVisitor<TSrc1, TSrc2, TDst, TContext>
        {
            Contracts.Check(aInput.Length == bInput.Length, "Vectors must have the same dimensionality.");

            // We handle the following cases:
            // 1. When a and b are both empty, we set the result to empty.
            // 2. When either a or b are dense, then the result will be dense, and we have some
            //    special casing for the sparsity of either a or b.
            // Then we have the case where both are sparse. We calculate the size of the output,
            // then fall through to the various cases.
            // 3. a and b have the same indices.
            // 4. a's indices are a subset of b's.
            // 5. b's indices are a subset of a's.
            // 6. Neither a nor b's indices are a subset of the other.
            VBuffer<TSrc1> localA = aInput;
            VBuffer<TSrc2> localB = bInput;

            if (localA.Count == 0 && localA.Count == 0)
            {
                // Case 1. Output will be empty.
                dst = new VBuffer<TDst>(localA.Length, 0, dst.Values, dst.Indices);
                return;
            }

            int aI = 0;
            int bI = 0;
            TDst[] values = dst.Values;
            if (localA.IsDense || localB.IsDense)
            {
                // Case 2. One of the two inputs is dense. The output will be dense.
                values = Utils.EnsureSize(values, localA.Length, localA.Length, keepOld: false);

                if (!localA.IsDense)
                {
                    // a is sparse, b is dense
                    for (int i = 0; i < localB.Length; i++)
                    {
                        TSrc1 aVal = (aI < localA.Count && i == localA.Indices[aI]) ? localA.Values[aI++] : default(TSrc1);
                        values[i] = visitor.Visit(i, aVal, localB.Values[i], ref context);
                    }
                }
                else if (!localB.IsDense)
                {
                    // b is sparse, a is dense
                    for (int i = 0; i < localA.Length; i++)
                    {
                        TSrc2 bVal = (bI < localB.Count && i == localB.Indices[bI]) ? localB.Values[bI++] : default(TSrc2);
                        values[i] = visitor.Visit(i, localA.Values[i], bVal, ref context);
                    }
                }
                else
                {
                    // both dense
                    for (int i = 0; i < localA.Length; i++)
                        values[i] = visitor.Visit(i, localA.Values[i], localB.Values[i], ref context);
                }
                dst = new VBuffer<TDst>(localA.Length, values, dst.Indices);
                return;
            }

            // a, b both sparse.
            int newCount = 0;
            while (aI < localA.Count && bI < localB.Count)
            {
                int aCompB = localA.Indices[aI] - localB.Indices[bI];
                if (aCompB <= 0) // a is no larger than b.
                    aI++;
                if (aCompB >= 0) // b is no larger than a.
                    bI++;
                newCount++;
            }

            if (aI < localA.Count)
                newCount += localA.Count - aI;
            if (bI < localB.Count)
                newCount += localB.Count - bI;

            // REVIEW: Worth optimizing the newCount == a.Length case?
            // Probably not...

            int[] indices = dst.Indices;
            indices = Utils.EnsureSize(indices, newCount, localA.Length, keepOld: false);
            values = Utils.EnsureSize(values, newCount, localA.Length, keepOld: false);

            if (newCount == localB.Count)
            {
                if (newCount == localA.Count)
                {
                    // Case 3, a and b actually have the same indices!
                    Array.Copy(localA.Indices, indices, localA.Count);
                    for (aI = 0; aI < localA.Count; aI++)
                    {
                        Contracts.Assert(localA.Indices[aI] == localB.Indices[aI]);
                        values[aI] = visitor.Visit(localA.Indices[aI], localA.Values[aI], localB.Values[aI], ref context);
                    }
                }
                else
                {
                    // Case 4, a's indices are a subset of b's.
                    Array.Copy(localB.Indices, indices, localB.Count);
                    aI = 0;
                    for (bI = 0; aI < localA.Count && bI < localB.Count; bI++)
                    {
                        Contracts.Assert(localA.Indices[aI] >= localB.Indices[bI]);
                        TSrc1 aVal = localA.Indices[aI] == localB.Indices[bI] ? localA.Values[aI++] : default(TSrc1);
                        values[bI] = visitor.Visit(localB.Indices[bI], aVal, localB.Values[bI], ref context);
                    }
                    for (; bI < localB.Count; bI++)
                        values[bI] = visitor.Visit(localB.Indices[bI], default(TSrc1), localB.Values[bI], ref context);
                }
            }
            else if (newCount == localA.Count)
            {
                // Case 5, b's indices are a subset of a's.
                Array.Copy(localA.Indices, indices, localA.Count);
                bI = 0;
                for (aI = 0; bI < localB.Count && aI < localA.Count; aI++)
                {
                    Contracts.Assert(localB.Indices[bI] >= localA.Indices[aI]);
                    TSrc2 bVal = localA.Indices[aI] == localB.Indices[bI] ? localB.Values[bI++] : default(TSrc2);
                    values[aI] = visitor.Visit(localA.Indices[aI], localA.Values[aI], bVal, ref context);
                }
                for (; aI < localA.Count; aI++)
                    values[aI] = visitor.Visit(localA.Indices[aI], localA.Values[aI], default(TSrc2), ref context);
            }
            else
            {
                // Case 6, neither a nor b's indices are a subset of the other.
                int newI = aI = bI = 0;
                TSrc1 aVal = default(TSrc1);
                TSrc2 bVal = default(TSrc2);
                while (aI < localA.Count && bI < localB.Count)
                {
                    int aCompB = localA.Indices[aI] - localB.Indices[bI];
                    int index = 0;

                    if (aCompB < 0)
                    {
                        index = localA.Indices[aI];
                        aVal = localA.Values[aI++];
                        bVal = default(TSrc2);
                    }
                    else if (aCompB > 0)
                    {
                        index = localB.Indices[bI];
                        aVal = default(TSrc1);
                        bVal = localB.Values[bI++];
                    }
                    else
                    {
                        index = localA.Indices[aI];
                        Contracts.Assert(index == localB.Indices[bI]);
                        aVal = localA.Values[aI++];
                        bVal = localB.Values[bI++];
                    }
                    values[newI] = visitor.Visit(index, aVal, bVal, ref context);
                    indices[newI++] = index;
                }

                for (; aI < localA.Count; aI++)
                {
                    int index = localA.Indices[aI];
                    values[newI] = visitor.Visit(index, localA.Values[aI], default(TSrc2), ref context);
                    indices[newI++] = index;
                }

                for (; bI < localB.Count; bI++)
                {
                    int index = localB.Indices[bI];
                    values[newI] = visitor.Visit(index, default(TSrc1), localB.Values[bI], ref context);
                    indices[newI++] = index;
                }
            }
            dst = new VBuffer<TDst>(localA.Length, newCount, values, indices);
        }

        /// <summary>
        /// Copy from a source list to the given VBuffer destination.
        /// </summary>
        public static void Copy<T>(List<T> src, ref VBuffer<T> dst, int length)
        {
            Contracts.CheckParam(0 <= length && length <= Utils.Size(src), nameof(length));
            var values = dst.Values;
            if (length > 0)
            {
                if (Utils.Size(values) < length)
                    values = new T[length];
                src.CopyTo(values);
            }
            dst = new VBuffer<T>(length, values, dst.Indices);
        }
    }
}
