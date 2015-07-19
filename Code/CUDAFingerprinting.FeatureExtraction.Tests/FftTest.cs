using System;
using System.Drawing;
using System.IO;
using CUDAFingerprinting.Common;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CUDAFingerprinting.FeatureExtraction.Tests
{
    [TestClass]
    public class FftTest
    {
       
        [TestMethod]
        public void FftTestMethod()
        {
            //Bitmap img = Fft.GenerateSinusoid();
            Bitmap img = Resources.test;
            int number = Math.Min(img.Width, img.Height);
            if (number % 2 == 1)
            {
                number--;
            }
            Bitmap resImg = new Bitmap(number, number);
            var arr = ImageHelper.LoadImageAsInt(img);
            double value;
            for (int i = -number/2; i < number/2; i++)
            {
                for (int j = -number/2; j < number/2; j++)
                {
                    value = Fft.GenerateAmplitudeSpectrum(i, j, arr, number);
                    resImg.SetPixel(i + number/2, j + number/2, Color.FromArgb(255-(int)(value/number/number) , 255-(int)(value/number/number) , 255-(int)(value/number/number)));
                }
                
            }
            int[,] amplitudeSpectrum = ImageHelper.LoadImageAsInt(resImg);
           // Fft.GradientMagnitudes(img);
            
            var dominant = Fft.FindDominantFrequency(amplitudeSpectrum);
            var k1 = dominant*number/2/Math.PI; // now k1 == k2
            int x = (int)number/(int)k1;

            resImg.Save(Path.GetTempPath() + Guid.NewGuid() + ".bmp");
            //Fft.GenerateAmplitudeSpectrum(10, 5, img);

        }
    }
}
