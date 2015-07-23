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
            var image = Resources.SampleFinger2;
            var array = ImageHelper.LoadImageAsInt(image);

            int wing = 8;

            RidgeLine newProssess = new RidgeLine(array, 2, wing);

            newProssess.NewSelection(210, 180);

            for (int i = 0; i < wing * 2 + 1; i++)
            {
                Console.Write("{0} ", newProssess._section[i]);
            }
            Console.WriteLine("\n{0}", array[210, 180]);
            Console.WriteLine("{0}", newProssess.FindNextMax());

            var newImage = new int[2 * wing + 1, 255];
            Array.Clear(newImage, 0, newImage.Length);

            for (int i = 0; i < 2 * wing + 1; i++)
            {
                int x = newProssess._section[i]/1000;
                int y = newProssess._section[i]%1000;

                for (int j = 0; j < array[x, y]; j++)
                {
                    newImage[i, j] = 120;
                }
            }

            ImageHelper.SaveArray(newImage, "testSections.bmp");
        }

        static void Main()
        {
            TestSection();
        }
    }
}
