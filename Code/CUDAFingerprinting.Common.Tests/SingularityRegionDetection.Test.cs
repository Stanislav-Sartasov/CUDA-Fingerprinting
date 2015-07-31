using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using CUDAFingerprinting.Common;
using System.Drawing;

using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CUDAFingerprinting.Common.Tests
{
    [TestClass]
    public class SingularityRegionDetectionTests
    {
        [TestMethod]
        public void SingularityRegionDetectionTest()
        {
            var image = Resources._8_2;
            var intBmp = ImageHelper.LoadImageAsInt(Resources._8_2);

            PixelwiseOrientationField field = new PixelwiseOrientationField(intBmp, 8);

            int width = intBmp.GetLength(0);
            int height = intBmp.GetLength(1);

            var orient = field.Orientation;

            System.IO.StreamWriter file = new System.IO.StreamWriter(@"D:\Orientation.txt");
            for (int i = 0; i < width; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    file.Write((int)(orient[i, j] * 10000));
                    file.Write(" ");
                }
                file.WriteLine();
            }

            for (int i = 0; i < width; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    orient[i, j] = (orient[i, j] * 10000) / 10000;
                }
            }

            SingularityRegionDetection D = new SingularityRegionDetection(orient);

            double[,] Result = D.Detect(orient);
            double[,] revertedResult = new double[height, width];
            
            for (int i = 0; i < width; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    revertedResult[height - 1 - j, i] = Result[i, j];
                }
            }

            Bitmap bmp = D.MakeBitmap(revertedResult);
            bmp.Save("Result.jpg");
        }
    }
}
