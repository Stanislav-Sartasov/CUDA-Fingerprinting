using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Drawing;

namespace CUDAFingerprinting.Common.Tests
{
    [TestClass]
    public class ImageHelperTests
    {
        [TestMethod]
        public void TestImageLoadAndSave()
        {
            var bmp = Resources.SampleFinger;
            var array = ImageHelper.LoadImage(bmp);

            for (int x = 0; x < bmp.Width; x++)
            {
                for (int y = 0; y < bmp.Height; y++)
                {
                    if ((double)bmp.GetPixel(x, y).R != array[bmp.Height - 1 - y, x])
                        Assert.Fail("Pixel is malformed at ({0}, {1})", x, y);
                }
            }

            var bmp2 = ImageHelper.SaveArrayToBitmap(array);

            for(int x=0;x<bmp2.Width;x++)
            {
                for(int y=0;y<bmp2.Height; y++)
                {
                    if (bmp.GetPixel(x, y) != bmp2.GetPixel(x, y))
                        Assert.Fail( "Pixels don't match at ({0}, {1})", x, y);
                }
            }
        }
    }
}
