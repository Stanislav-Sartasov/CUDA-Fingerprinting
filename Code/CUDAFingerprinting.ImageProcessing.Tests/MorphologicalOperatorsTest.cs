using System.Diagnostics;
using System.IO;
using CUDAFingerprinting.Common;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CUDAFingerprinting.ImageProcessing.Tests
{
    [TestClass]
    public class MorphologicalOperatorsTest
    {
        public static int[] Binarization(int[] a)
        {
            const int BINARIZATION_BARIER = 128;
            for (int i = 0; i < a.Length; i++)
            {
                a[i] = a[i] > BINARIZATION_BARIER ? 255 : 0;
            }
            return a;
        }

        private static int BLACK = MorphologicalOperators.BLACK;
        private static int WHITE = MorphologicalOperators.WHITE;

        private int[] structElem = new int[]{
                WHITE, BLACK, WHITE, 
                BLACK, BLACK, BLACK, 
                WHITE, BLACK, WHITE
            };

        [TestMethod]
        public void ErosionTest()
        {
            var bmp = Resources.f;
            var img = ImageHelper.LoadImageAsInt(bmp);

            int h = img.GetLength(0);
            int w = img.GetLength(1);

            var erosed = ImageHelper.SaveArrayToBitmap(
                Array1Dto2D(
                    MorphologicalOperators.Erosion(
                        Binarization(Array2Dto1D(img)), 
                        structElem, 
                        w, h, 
                        2, 2),
                    w,
                    h
                )
            );
            var name = Path.GetTempPath() + bmp.GetHashCode().ToString() + "Erosion.bmp";
            erosed.Save(name, ImageHelper.GetImageFormatFromExtension(name));
            Process.Start(name);
        }

        [TestMethod]
        public void DilationTest()
        {
            var bmp = Resources.f;
            var img = ImageHelper.LoadImageAsInt(bmp);

            int h = img.GetLength(0);
            int w = img.GetLength(1);

            var dilated = ImageHelper.SaveArrayToBitmap(
                Array1Dto2D(
                    MorphologicalOperators.Dilation(
                        Binarization(Array2Dto1D(img)), 
                        structElem, 
                        w, h, 
                        2, 2),
                    w,
                    h
                )
            );
            var name = Path.GetTempPath() + bmp.GetHashCode().ToString() + "Dilation.bmp";
            dilated.Save(name, ImageHelper.GetImageFormatFromExtension(name));
            Process.Start(name);
        }

        [TestMethod]
        public void OpeningTest()
        {
            var bmp = Resources.f;
            var img = ImageHelper.LoadImageAsInt(bmp);

            int h = img.GetLength(0);
            int w = img.GetLength(1);

            var opened = ImageHelper.SaveArrayToBitmap(
                Array1Dto2D(
                    MorphologicalOperators.Opening(
                        Binarization(Array2Dto1D(img)),
                        structElem,
                        w, h,
                        2, 2),
                    w,
                    h
                )
            );
            var name = Path.GetTempPath() + bmp.GetHashCode().ToString() + "Opening.bmp";
            opened.Save(name, ImageHelper.GetImageFormatFromExtension(name));
            Process.Start(name);
        }

        [TestMethod]
        public void ClosingTest()
        {
            var bmp = Resources.f;
            var img = ImageHelper.LoadImageAsInt(bmp);

            int h = img.GetLength(0);
            int w = img.GetLength(1);

            var closed = ImageHelper.SaveArrayToBitmap(
                Array1Dto2D(
                    MorphologicalOperators.Closing(
                        Binarization(Array2Dto1D(img)),
                        structElem,
                        w, h,
                        2, 2),
                    w,
                    h
                )
            );
            var name = Path.GetTempPath() + bmp.GetHashCode().ToString() + "Closing.bmp";
            closed.Save(name, ImageHelper.GetImageFormatFromExtension(name));
            Process.Start(name);
        }

        [TestMethod]
        public void ComposeTest()
        {
            var bmp = Resources.f;
            var img = ImageHelper.LoadImageAsInt(bmp);

            int h = img.GetLength(0);
            int w = img.GetLength(1);

            var closed = MorphologicalOperators.Closing(
                Binarization(Array2Dto1D(img)), 
                structElem, 
                w, h, 
                2, 2);

            var erosed = ImageHelper.SaveArrayToBitmap(
                Array1Dto2D(
                    MorphologicalOperators.Opening(
                        closed, 
                        structElem, 
                        w, h, 
                        2, 2),
                    w,
                    h
                )
            );
            var name = Path.GetTempPath() + bmp.GetHashCode().ToString() + "Compose.bmp";
            erosed.Save(name, ImageHelper.GetImageFormatFromExtension(name));
            Process.Start(name);
        }

        private static int[] Array2Dto1D(int[,] data)
        {
            int h = data.GetLength(0);
            int w = data.GetLength(1);
            int[] result = new int[h * w];
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    result[(h - 1 - y) * w + x] = data[h - 1 - y, x];
                }
            }
            return result;
        }

        private static int[,] Array1Dto2D(int[] data, int w, int h)
        {
            int[,] result = new int[h, w];
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    result[(h - 1 - y), x] = data[(h - 1 - y) * w + x];
                }
            }
            return result;
        }

    }
}