using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Drawing;
using System.IO;
using CUDAFingerprinting.Common;
using System.Diagnostics;

namespace CUDAFingerprinting.Common.Thinning.Tests
{
    [TestClass]
    public class ThinnerTests
    {
        [TestMethod]
        public void ThinTestAllPics()
        {
            //TestThin(Resources.patt);
            //TestThin(Resources.line);
            //TestThin(Resources.ManPic);
            //TestThin(Resources.small);
            //TestThin(Resources.verySmall);
            //TestThin(Resources.verySmall2);
            //TestThin(Resources._101_8);//very bad fingerprint
            //TestThin(Resources._19_4);
            //TestThin(Resources._29_4);
            //TestThin(Resources.X);
            //TestThin(Resources.connectedness);
            //TestThin(Resources.connectedness2);
            //TestThin(Resources.idealH);
            TestThin(Resources._1_1);
        }

        //BINARIZATION_BARIER increasing <=> less loss of information, more noise
        public static double[,] Binarization(double[,] a, int h, int w)
        {
            const double BINARIZATION_BARIER = 128.0;
            double[,] nA = new double[h, w];
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    nA[y, x] = a[y, x] < BINARIZATION_BARIER ? 0.0 : 255.0;
            return nA;
        }

        public void TestThin(Bitmap bmp)
        {
            var bmpBefore = Binarization(ImageHelper.LoadImage(bmp), bmp.Height, bmp.Width);
            /* open binarized source picture
            var newPic1 = ImageHelper.SaveArrayToBitmap(bmpBefore);
            var name1 = Path.GetTempPath() + bmp.GetHashCode().ToString() + "BEFORE.bmp";
            newPic1.Save(name1, ImageHelper.GetImageFormatFromExtension(name1));
            Process.Start(name1);
            */
            var bmpAfter = Thinner.Thin(bmpBefore, bmp.Width, bmp.Height);
            
            var newPic = ImageHelper.SaveArrayToBitmap(
                OverlapArrays(bmpAfter, bmpBefore, bmp.Height, bmp.Width)
            );
            var name = Path.GetTempPath() + bmp.GetHashCode().ToString() + "AFTER.bmp";
            newPic.Save(name, ImageHelper.GetImageFormatFromExtension(name));
            Process.Start(name);
        }

        //overlaps skeleton above background
        public static double[,] OverlapArrays(double[,] skeleton, double[,] background, int h, int w)
        {
            double[,] nA = new double[h, w];
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    nA[y, x] = skeleton[y, x] < 250.0 ? 128.0 : background[y, x];
            return nA;
        }
        /*
        public static void WriteArrayPic(String s, double[,] bytes, int h, int w)
        {
            System.Console.WriteLine(s);
            for (int x = 0; x < h; x++)
            {
                for (int y = 0; y < w; y++)
                {
                    System.Console.Write(bytes[h - 1 - x, y] != 0 ? '=' : '+');
                }
                System.Console.WriteLine();
            }
        }*/
    }
}
