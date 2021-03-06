﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CUDAFingerprinting.FeatureExtraction.TemplateCreate
{
    public class Cylinder3D
    {
        public uint[] Cylinder { get; set; }

        public Cylinder3D()
        {
            Cylinder = new uint[TemplateCreator.NumberCell / 32];
        }

        public void SetValue(int i, int j, int k, byte value)
        {
            Cylinder[Linearization(i, j, k) / 32] |= ((uint)value << Linearization(i, j, k) % 32);
        }

        public byte GetValue(int i, int j, int k)
        {
            return (byte)((Cylinder[Linearization(i, j, k) / 32] & ((uint)1 << Linearization(i, j, k) % 32)) >> Linearization(i, j, k) % 32);
        }

        private int Linearization(int i, int j, int k)
        {
            return (k - 1) * TemplateCreator.BaseCuboid * TemplateCreator.BaseCuboid +
                (j - 1) * TemplateCreator.BaseCuboid + i - 1;
        }
    }
}
