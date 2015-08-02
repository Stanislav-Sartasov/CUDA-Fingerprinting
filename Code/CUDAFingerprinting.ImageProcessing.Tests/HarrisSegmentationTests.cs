using CUDAFingerprinting.ImageProcessing.Segmentation;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CUDAFingerprinting.ImageProcessing.Tests
{
    [TestClass]
    public class HarrisSegmentationTests
    {
        [TestMethod]
        public void HarrisSegmentationTest()
        {
            var image = Resources.SampleFinger;
            HarrisSegmentation M = new HarrisSegmentation(image);

            double[,] matrix = M.GaussFilter();
            int[,] byteMatrix = M.Segmentate(matrix);

            string filename = "Result.jpg";

            var bmp = M.MakeBitmap(byteMatrix);
            M.SaveSegmentation(bmp, filename);

            image.Dispose();
            bmp.Dispose();
        }
    }
}
