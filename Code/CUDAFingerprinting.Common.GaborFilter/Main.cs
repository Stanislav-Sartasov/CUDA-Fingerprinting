using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
namespace CUDAFingerprinting.Common.GaborFilter
{
    class Program
    {
        private static void Main(string[] args)
        {
            //var mas = new GaborFilter(8, 5);

            //mas.Filters[2].WriteMatrix();
            var bmp = Resources.SampleFinger1;
            double[,] imgDoubles = ImageHelper.LoadImage(bmp);
            int[,] imgInts = ImageHelper.LoadImageAsInt(bmp);
            OrientationField.OrientationField orf = new OrientationField.OrientationField(imgInts, 16);
            double[,] orient = orf.GetOrientationMatrix(imgInts.GetLength(0), imgInts.GetLength(1));
            var res = ImageEnhancement.Enhance(imgDoubles, orient, (double) 1/9, 5, 8);
            var bmp2 = ImageHelper.SaveArrayToBitmap(res);
            bmp2.Save("test.bmp", ImageHelper.GetImageFormatFromExtension("test.bmp"));
        }
    }
}
