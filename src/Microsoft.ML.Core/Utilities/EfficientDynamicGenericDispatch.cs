// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Runtime.CompilerServices;
using System.Reflection;

public class DynamicTypeInvoker<TDelegate> where TDelegate : class
{
    public DynamicTypeInvoker(int validTypeArrayLen, Func<Type[], TDelegate> resolver)
    {
        _lock = new object();
        _cache = new Entry[1];
        _resolver = resolver;
        _typeArrayLen = validTypeArrayLen;
    }

    public DynamicTypeInvoker(int validTypeArrayLen, Func<Type[], MethodInfo> resolver) : this(validTypeArrayLen, (Type[] types) => { return (TDelegate)(object)resolver(types).CreateDelegate(typeof(TDelegate)); })
    {
    }

    private class Entry
    {
        public Type[] Types;
        public TDelegate Result;
        public Entry Next;
    }

    // Initialize the cache eagerly to avoid null checks.
    // Use array with just single element to make this pay-for-play. The actual cache will be allocated only
    // once the lazy lookups are actually needed.
    private Entry[] _cache;

    private readonly object _lock;
    private readonly int _typeArrayLen;

    private Func<Type[], TDelegate> _resolver;

    public TDelegate GetDelegate(Type type)
    {
        Entry entry = LookupInCache(_cache, type);
        if (entry == null)
        {
            entry = CacheMiss(type);
        }
        return entry.Result;
    }

    [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
    private static Entry LookupInCache(Entry[] cache, Type type)
    {
        int key = 653465;

        key = ((key >> 4) ^ type.GetHashCode());

        key &= (cache.Length - 1);
        Entry entry = cache[key];
        while (entry != null)
        {
            if (type == entry.Types[0])
                break;

            entry = entry.Next;
        }
        return entry;
    }

    [MethodImplAttribute(MethodImplOptions.NoInlining)]
    private unsafe Entry CacheMiss(Type type)
    {
        return CacheMiss(new Type[] { type });
    }

    public TDelegate GetDelegate(Type type, Type type2)
    {
        Entry entry = LookupInCache(_cache, type, type2);
        if (entry == null)
        {
            entry = CacheMiss(type, type2);
        }
        return entry.Result;
    }

    [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
    private static Entry LookupInCache(Entry[] cache, Type type, Type type2)
    {
        int key = 653465;

        key = ((key >> 4) ^ type.GetHashCode());
        key = ((key >> 4) ^ type2.GetHashCode());

        key &= (cache.Length - 1);
        Entry entry = cache[key];
        while (entry != null)
        {
            if ((type == entry.Types[0]) && (type2 == entry.Types[1]))
                break;

            entry = entry.Next;
        }
        return entry;
    }

    [MethodImplAttribute(MethodImplOptions.NoInlining)]
    private unsafe Entry CacheMiss(Type type, Type type2)
    {
        return CacheMiss(new Type[] { type, type2 });
    }

    public TDelegate GetDelegate(params Type[] types)
    {
        Entry entry = LookupInCache(_cache, types);
        if (entry == null)
        {
            entry = CacheMiss(types);
        }
        return entry.Result;
    }

    [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
    private static Entry LookupInCache(Entry[] cache, Type[] types)
    {
        int key = 653465;

        foreach (Type t in types)
        {
            key = ((key >> 4) ^ t.GetHashCode());
        }

        key &= (cache.Length - 1);
        Entry entry = cache[key];
        while (entry != null)
        {
            bool mismatchFound = false;
            for (int i = 0; i < types.Length; i++)
                if (types[i] != entry.Types[i])
                    continue;

            if (!mismatchFound)
                break;

            entry = entry.Next;
        }
        return entry;
    }

    [MethodImplAttribute(MethodImplOptions.NoInlining)]
    private unsafe Entry CacheMiss(Type[] types)
    {
        if (types.Length != _typeArrayLen)
            throw new IndexOutOfRangeException();

        TDelegate result = _resolver(types);

        lock(_lock)
        {
            // Avoid duplicate entries
            Entry existingEntry = LookupInCache(_cache, types);
            if (existingEntry != null)
                return existingEntry;

            // Resize cache as necessary
            Entry[] cache = ResizeCacheForNewEntryAsNecessary();

            int key = 653465;

            foreach (Type t in types)
            {
                key = ((key >> 4) ^ t.GetHashCode());
            }

            key &= (cache.Length - 1);

            Entry newEntry = new Entry() { Result = result, Types = types, Next = cache[key] };
            cache[key] = newEntry;
            return newEntry;
        }
    }

    //
    // Parameters and state used by generic lookup cache resizing algorithm
    //

    private const int InitialCacheSize = 16; // MUST BE A POWER OF TWO
    private const int DefaultCacheSize = 128;
    private const int MaximumCacheSize = 1024;

    private int _tickCountOfLastOverflow;
    private int _entries;
    private bool _roundRobinFlushing;

    private Entry[] ResizeCacheForNewEntryAsNecessary()
    {
        Entry[] cache = _cache;

        if (cache.Length < InitialCacheSize)
        {
            // Start with small cache size so that the cache entries used by startup one-time only initialization will get flushed soon
            return _cache = new Entry[InitialCacheSize];
        }

        int entries = _entries++;

        // If the cache has spare space, we are done
        if (2 * entries < cache.Length)
        {
            if (_roundRobinFlushing)
            {
                cache[2 * entries] = null;
                cache[2 * entries + 1] = null;
            }
            return cache;
        }

        //
        // Now, we have cache that is overflowing with the stuff. We need to decide whether to resize it or start flushing the old entries instead
        //

        // Start over counting the entries
        _entries = 0;

        // See how long it has been since the last time the cache was overflowing
        int tickCount = Environment.TickCount;
        int tickCountSinceLastOverflow = tickCount - _tickCountOfLastOverflow;
        _tickCountOfLastOverflow = tickCount;

        bool shrinkCache = false;
        bool growCache = false;

        if (cache.Length < DefaultCacheSize)
        {
            // If the cache have not reached the default size, just grow it without thinking about it much
            growCache = true;
        }
        else
        {
            if (tickCountSinceLastOverflow < cache.Length / 128)
            {
                // If the fill rate of the cache is faster than ~0.01ms per entry, grow it
                if (cache.Length < MaximumCacheSize)
                    growCache = true;
            }
            else
            if (tickCountSinceLastOverflow > cache.Length * 16)
            {
                // If the fill rate of the cache is slower than 16ms per entry, shrink it
                if (cache.Length > DefaultCacheSize)
                    shrinkCache = true;
            }
            // Otherwise, keep the current size and just keep flushing the entries round robin
        }

        if (growCache || shrinkCache)
        {
            _roundRobinFlushing = false;

            return _cache = new Entry[shrinkCache ? (cache.Length / 2) : (cache.Length * 2)];
        }
        else
        {
            _roundRobinFlushing = true;
            return cache;
        }
    }
}
