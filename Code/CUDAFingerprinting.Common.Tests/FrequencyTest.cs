using System;
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
            //var bmp2 = ImageHelper.SaveArrayToBitmap(ar2, true);
            //bmp2.Save("003.bmp", ImageHelper.GetImageFormatFromExtension("003.bmp"));

            var fr = LocalRidgeFrequency.CalculateFrequency(ar2, orMatr);

            int ncount = 0;
            for (int i = 0; i < fr.GetLength(0); i++)
                for (int j = 0; j < fr.GetLength(1); j++)
                {
                    if ((fr[i, j] == -1.0))
                        ncount++;
                }
            int pixNum = fr.GetLength(0) * fr.GetLength(1);

            fr.InterpolateToPerfecton();
            
            var filtered = LocalRidgeFrequency.FilterFrequencies(fr);

            double sum = 0;
            for (int i = 0; i < fr.GetLength(0); i++)
                for (int j = 0; j < fr.GetLength(1); j++)
                {
                    sum += fr[i, j];
                }
            double mean = sum / pixNum;

            int count = 0;
            for (int i=0; i < filtered.GetLength(0); i++)
                for (int j = 0; j < filtered.GetLength(1); j++)
                {
                    if ((filtered[i, j] == -1.0) || (filtered[i, j] != filtered[i, j]))
                        count ++;
                }
        }
        [TestMethod]
        public void TestGaussian()
        {
            var gaussian = new Filter(7, 0.84089642);
            gaussian.Normalize();
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
