using System;
using System.Collections.Generic;
using System.Linq;

namespace CUDAFingerprinting.Common
{
    public struct Descriptor
    {
        public List<Minutia> Minutias;
        public Minutia Center;
        public Descriptor(List<Minutia> m, Minutia c)
        {
            Minutias = m.ToList();
            Center = c;
        }
    };
}