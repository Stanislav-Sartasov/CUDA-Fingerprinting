using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;

namespace CUDAFingerprinting.Common
{
//    [StructLayout(LayoutKind.Sequential)]
    public struct Descriptor
    {
//        [MarshalAs(UnmanagedType.LPArray)]
        public List<Minutia> Minutias;
//        [MarshalAs(UnmanagedType.Struct)]
        public Minutia Center;
        public Descriptor(List<Minutia> m, Minutia c)
        {
            Minutias = m.ToList();
            Center = c;
        }
    };
}