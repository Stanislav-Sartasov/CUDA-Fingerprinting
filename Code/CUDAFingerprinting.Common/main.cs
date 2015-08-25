using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using CUDAFingerprinting.Common.OrientationField;

namespace CUDAFingerprinting.Common
{
    class main
    {
        static void Main(string[] args)
        {
            var bytes = ImageHelper.LoadImage<int>("D:\\SmrSchl\\DB2_bmp\\DB2_bmp\\1_5.bmp");
            
            PixelwiseOrientationField field = new PixelwiseOrientationField(bytes, 16);

            SmoothOrientationField SO_field = new SmoothOrientationField(field.Orientation);
            field.NewOrientation(SO_field.LocalOrientation());
  

          //  Filter f = new Filter(16, 2.5);
          //  f.WriteMatrix();
           

           // Console.Read();
        }
    }
}
