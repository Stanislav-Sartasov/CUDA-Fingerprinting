using System;
using System.Runtime.InteropServices;

namespace CUDAFingerprinting.Common
{
    [StructLayout(LayoutKind.Sequential)]
    public struct Minutia
    {
        [MarshalAs(UnmanagedType.R4)]
        public Single Angle;
        [MarshalAs(UnmanagedType.I4)]
        public Int32 X;
        [MarshalAs(UnmanagedType.I4)]
        public Int32 Y;
    };
}
