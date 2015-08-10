using System;

namespace CUDAFingerprinting.Common
{
    public struct Descriptor
    {
        public Minutia[] Minutias;
        public Minutia Center;
        public Descriptor(Minutia[] m, Minutia c)
        {
            this.Minutias = m;
            this.Center = c;
        }
    };
}

