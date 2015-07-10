using System;
using CUDAFingerprinting.Common;
using CUDAFingerprinting.Common.OrientationField;
using CUDAFingerprinting.Common.GaborFilter;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace ImageEnhancmentTest
{
    [TestClass]
    public class EnhancmentTest
    {
        [TestMethod]
        public void ImageEnhancmentTest1()
        {
            var bmp = Resources.SampleFinger;
            double[,] imgDoubles = ImageHelper.LoadImage(bmp);
            imgDoubles.DoNormalization(100, 100);
            int[,] imgInts = imgDoubles.Select2D((x => (int)x));
            OrientationField orf = new OrientationField(imgInts, 16);
            double[,] orient = orf.GetOrientationMatrix(imgInts.GetLength(0), imgInts.GetLength(1));
            var res = ImageEnhancement.Enhance(imgDoubles, orient, (double)1 / 9, 32, 8);
            var bmp2 = ImageHelper.SaveArrayToBitmap(res);
            bmp2.Save("test.bmp", ImageHelper.GetImageFormatFromExtension("test.bmp"));
        }
    }
}
