using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CUDAFingerprinting.Common.Tests
{
    [TestClass]
    public class FrequencyTest
    {
        [TestMethod]
        public void TestFrequency()
        {
            var bmp = Resources.SampleFinger3;
            var array = ImageHelper.LoadImageAsInt(bmp);

            var orfield = new OrientationField(array);
            var orMatr = orfield.GetOrientationMatrix(array.GetLength(0), array.GetLength(1));
            var ar2 = array.Select2D(x => (double) x).DoNormalization(100, 100);
            var freq = LocalRidgeFrequency.CalculateFrequency(ar2, orMatr);
            int count = 0;
            for (int i=0; i < freq.GetLength(0); i++)
                for (int j = 0; j < freq.GetLength(1); j++)
                {
                    if ((freq[i, j] == -1.0) || (freq[i, j] > 0.3333333) || ((freq[i, j] < 0.04) && (freq[i, j] != 0)))
                        count ++;
                }
            int a = freq.GetLength(0)*freq.GetLength(1);
        }
        [TestMethod]
        public void TestGaussian()
        {
            var gaussian = new Filter(7,0.0865);
            var mean = gaussian.Matrix.CalculateMean();
            var variation = gaussian.Matrix.CalculateVariation(mean);
        }
    }
}
