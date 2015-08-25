using CUDAFingerprinting.ImageProcessing.Segmentation;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Drawing;

namespace CUDAFingerprinting.FeatureExtraction.Tests
{
    [TestClass]
    public class SegmentatatorTests
    {
        [TestMethod]
        public void SegmentatorTest()
        {
            var image = Resources.SampleFinger;
            Segmentator M = new Segmentator(image);

            float [,] matrix = M.SobelFilter();
            byte[,] byteMatrix = M.Segmentate();

            string filename = "Result.jpg";

            Bitmap bmp = M.MakeBitmap(byteMatrix);
            M.SaveSegmentation(bmp, filename);

            image.Dispose();
            bmp.Dispose();
        }
    }
}