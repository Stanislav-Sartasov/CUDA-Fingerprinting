using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
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
            var bmp = Resources.SampleFinger5;
            var array = ImageHelper.LoadImageAsInt(bmp);

            var orfield = new OrientationField(array);
            var orMatr = orfield.GetOrientationMatrix(array.GetLength(0), array.GetLength(1));
            var ar2 = array.Select2D(x => (double)x).DoNormalization(100, 100); 
            var bmp2 = ImageHelper.SaveArrayToBitmap(ar2, true);
            bmp2.Save("003.bmp", ImageHelper.GetImageFormatFromExtension("003.bmp"));
            var fr = LocalRidgeFrequency.CalculateFrequency(ar2, orMatr);

            int ncount = 0;
            for (int i = 0; i < fr.GetLength(0); i++)
                for (int j = 0; j < fr.GetLength(1); j++)
                {
                    if ((fr[i, j] == -1.0))
                        ncount++;
                }
            int aa = fr.GetLength(0) * fr.GetLength(1);

            fr.InterpolateToPerfecton();
            int k = 0;
            double sum = 0;
            for (int i=0; i < fr.GetLength(0); i++)
                for (int j = 0; j < fr.GetLength(1); j ++)
                {
                    k++;
                    sum += fr[i, j];
                }
            double mean = sum/k;
            var filtered = LocalRidgeFrequency.Filter(fr, 7, 1);

            int count2 = 0;
            for (int i = 0; i < fr.GetLength(0); i++)
                for (int j = 0; j < fr.GetLength(1); j++)
                {
                    if (Math.Abs(fr[i, j] - filtered[i, j]) > 0)
                    {
                        double a1 = fr[i, j];
                        double a2 = filtered[i, j];
                        if (a1 < a2)
                        {
                            count2++;
                        }
                       // count2++;
                    }
                }
            //for (int i = 0; i<1; i++)
            //    freq = LocalRidgeFrequency.InterpolateFrequency(freq, array.GetLength(0), array.GetLength(1));
            var freq = LocalRidgeFrequency.GetFrequencyMatrixImageSize(filtered, array.GetLength(0), array.GetLength(1));
            int count = 0;
            for (int i=0; i < freq.GetLength(0); i++)
                for (int j = 0; j < freq.GetLength(1); j++)
                {
                    if ((freq[i, j] == -1.0) || (freq[i, j] > 1.0 / 3.0) || ((freq[i, j] < 0.04) && (freq[i, j] != 0)) || (freq[i, j] != freq[i, j]))
                        count ++;
                }
            int a = freq.GetLength(0)*freq.GetLength(1);
        }
        [TestMethod]
        public void TestGaussian()
        {
            var gaussian = new Filter(19, 3);
           // gaussian.Normalize();
            double sum = 0;
            for (int i = 0; i < gaussian.Matrix.GetLength(0); i++)
                for (int j = 0; j < gaussian.Matrix.GetLength(1); j++)
                {
                    sum += gaussian.Matrix[i, j];
                }
            var mean = gaussian.Matrix.CalculateMean();
            var variation = gaussian.Matrix.CalculateVariation(mean);
        }
    }
}
