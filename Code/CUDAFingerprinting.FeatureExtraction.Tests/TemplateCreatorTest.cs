using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CUDAFingerprinting.Common;
using CUDAFingerprinting.Common.OrientationField;
using CUDAFingerprinting.ImageProcessing.GaborEnhancement;
using CUDAFingerprinting.ImageProcessing.Postprocessing;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CUDAFingerprinting.FeatureExtraction.Tests
{
    [TestClass]
    public class TemplateCreatorTest
    {
        private int[] Array2Dto1D(int[,] data)
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

        [TestMethod]
        public void CreatTemplateTest()
        {
            /* double[,] img1 = ImageHelper.LoadImage<double>(Resources._1_1);
             double[,] img2 = ImageHelper.LoadImage<double>(Resources._1_2*/
            double[,] imgDoubles = ImageHelper.LoadImage<double>(Resources._1_1);

            imgDoubles.DoNormalization(100, 100);

            int[,] imgInts = imgDoubles.Select2D((x => (int)x));
            OrientationField orf = new OrientationField(imgInts, 16);
            double[,] orient = orf.GetOrientationMatrix(imgInts.GetLength(0), imgInts.GetLength(1));

            var freqMatrx = LocalRidgeFrequency.GetFrequencies(imgDoubles, orient);

            var res = ImageEnhancement.Enhance(imgDoubles, orient, freqMatrx, 32, 8);


           /* var img = ImageHelper.LoadImage<int>(Resources._1_1);

            int h = img.GetLength(0);
            int w = img.GetLength(1);

            int[] withoutHolesAndIslands = HolesAndIslandsResolver.ResolveHolesAndIslands(
                Array2Dto1D(ImageProcessing.Binarization.ImageBinarization.Binarize2D(img, 128)),
                16,
                9,
                w, h);*/
        }
    }
}
