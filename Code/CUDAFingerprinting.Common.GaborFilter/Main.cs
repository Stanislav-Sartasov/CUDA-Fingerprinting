using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using System.IO;
using CUDAFingerprinting.Common.OrientationField;

namespace CUDAFingerprinting.Common.GaborFilter
{
    class Program
    {
        private static void Main(string[] args)
        {
            var mas = new GaborFilter(8, 15, (double) 1/9);

            //for (int i = 0; i < 8; i++)
            //{
            //    ImageHelper.SaveArray(mas.Filters[i].Matrix, "Filter" + Convert.ToString(i) + ".bmp");
            //}

            //mas.Filters[2].WriteMatrix();
            var bmp = Resources.SampleFinger1;
            double[,] imgDoubles = ImageHelper.LoadImage(bmp);
            //int[,] imgInts = ImageHelper.LoadImageAsInt(bmp);
            imgDoubles.DoNormalization(100, 100);
            var bmp001 = ImageHelper.SaveArrayToBitmap(imgDoubles);
            bmp001.Save("test1.bmp", ImageHelper.GetImageFormatFromExtension("test1.bmp"));
            int[,] imgInts = ImageHelper.LoadImageAsInt(bmp001);
            //bmp001.Save("test1.bmp", ImageHelper.GetImageFormatFromExtension("test1.bmp"));
            //var bmp002 = ImageHelper.SaveArrayToBitmap(imgInts);
            //bmp002.Save("test2.bmp", ImageHelper.GetImageFormatFromExtension("test2.bmp"));
            OrientationField.OrientationField orf = new OrientationField.OrientationField(imgInts, 16);
          //  orf.SaveToFile(Path.GetTempPath() + Guid.NewGuid() + ".bmp", true);
            double[,] orient = orf.GetOrientationMatrix(imgInts.GetLength(0), imgInts.GetLength(1));
            var res = ImageEnhancement.Enhance(imgDoubles, orient, (double)1 / 9, 51, 8);
            var bmp2 = ImageHelper.SaveArrayToBitmap(res);
            bmp2.Save("test.bmp", ImageHelper.GetImageFormatFromExtension("test.bmp"));
        }
    }
}
