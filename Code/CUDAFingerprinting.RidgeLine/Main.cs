using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CUDAFingerprinting.Common;

namespace CUDAFingerprinting.RidgeLine
{
    class Program
    {
        static void TestSection()
        {
            var image = Resources.SampleFinger3;
            var array = ImageHelper.LoadImageAsInt(image);

            int wing = 8;

            RidgeLine newProssess = new RidgeLine(array, 10, wing);

            newProssess.GoToLine();

            //var newImage = new int[2 * wing + 1, 255];
            //Array.Clear(newImage, 0, newImage.Length);

            //for (int i = 0; i < 2 * wing + 1; i++)
            //{
            //    int x = newProssess._section[i]/1000;
            //    int y = newProssess._section[i]%1000;

            //    for (int j = 0; j < array[x, y]; j++)
            //    {
            //        newImage[i, j] = 120;
            //    }
            //}

            //ImageHelper.SaveArray(newImage, "testSections.bmp");
        }

        static void Main()
        {
            TestSection();
        }
    }
}
