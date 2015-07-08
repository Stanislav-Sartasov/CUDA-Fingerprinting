using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CUDAFingerprinting.Common.Tests
{
    [TestClass]
    public class NormalizeTests
    {
        [TestMethod]
        public void TestNormalization()
        {
            var bmp = Resources.SampleFinger3;
            var array = ImageHelper.LoadImage(bmp);

            array = array.DoNormalization(100, 1000);

            var bmp2 = ImageHelper.SaveArrayToBitmap(array);

            bmp2.Save("test.bmp", ImageHelper.GetImageFormatFromExtension("test.bmp"));
        }
    }
}