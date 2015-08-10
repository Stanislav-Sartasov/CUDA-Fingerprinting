using System;
using System.Collections.Generic;

namespace CUDAFingerprinting.Common
{
    public struct Descriptor
    {
        public List<Minutia> Minutias;
        public Minutia Center;
        public Descriptor(List<Minutia> m, Minutia c)
        {
            Minutias = m;
            Center = c;
        }
    };
}

