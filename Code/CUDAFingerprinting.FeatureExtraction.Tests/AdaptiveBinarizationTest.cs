using System;
using System.Drawing;
using System.Globalization;
using System.IO;
using CUDAFingerprinting.Common;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CUDAFingerprinting.FeatureExtraction.Tests
{
    [TestClass]
    public class AdaptiveBinarizationTest
    {
        [TestMethod]
        public void AdaptiveBinarizationTestMethod()
        {
            int[,] arrayI = ImageHelper.LoadImageAsInt(Resources.test);
            int x = 44, y = 14;
            var proj = AdaptiveBinarization.ProjectionX(x, y, arrayI);
            int columnWidth = 20;
            int bmpHeight = 256;
            int projectionLength = 16;
            Bitmap barChart = new Bitmap(columnWidth * projectionLength, bmpHeight);

            for (int i = 0; i < columnWidth * projectionLength; i += columnWidth)
            {
                for (int j = bmpHeight-1; j > bmpHeight - proj[i/columnWidth]; j--)
                {
                    for (int k = 0; k < columnWidth; k++)
                    {
                        barChart.SetPixel(i + k, j, Color.FromArgb(proj[i / columnWidth], proj[i / columnWidth], proj[i / columnWidth]));
                    }
                }
                Graphics graphics = Graphics.FromImage(barChart);
                graphics.DrawString(proj[i / columnWidth].ToString(CultureInfo.InvariantCulture), new Font("Times New Roman", 9), new SolidBrush(Color.Crimson), new PointF(i + 0.1f, (float) (bmpHeight - 20))); 
            }
            barChart.Save(Path.GetTempPath() + Guid.NewGuid() + ".bmp");
        }
    }
}
