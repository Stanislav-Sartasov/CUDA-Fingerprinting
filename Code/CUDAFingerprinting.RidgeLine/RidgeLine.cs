using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CUDAFingerprinting.Common;

namespace CUDAFingerprinting.RidgeLine
{
    class RidgeLine
    {
        private int[,] _image;
        private OrientationField _orientation;
        private int _step;
        private int _wings;
        private const int BuildUp = 1000;  //specially for trade coordinates between methods

        public int[] _section;

        public RidgeLine(int[,] image, int step, int wings)
        {
            _image = image;
            _orientation = new OrientationField(image);
            _step = step;
            _wings = wings;
            _section = new int[2*wings + 1];
        }

        public void NewSelection(int mid)
        {
            int i = mid/BuildUp;
            int j = mid%BuildUp;

            double angle = _orientation.GetOrientation(i, j) + Math.PI / 2;

            _section[_wings] = i * BuildUp + j;

            for (var k = 1; k <= _wings; k++)
            {
                int xs, ys, xe, ye;
                xs = Convert.ToInt32(i - k * Math.Cos(angle));
                ys = Convert.ToInt32(j - k * Math.Sin(angle));
                xe = Convert.ToInt32(i + k * Math.Cos(angle));
                ye = Convert.ToInt32(j + k * Math.Sin(angle));

                _section[_wings - k] = xs * BuildUp + ys;
                _section[_wings + k] = xe * BuildUp + ye;
            }
        }

        public int FindNextMax()
        {
            int loc = 0;
            int i = 1;

            while (loc == 0)
            {
                int x = _section[_wings - i] / BuildUp;
                int y = _section[_wings - i] % BuildUp;

                if (_image[x, y] == 255)
                {
                    loc = _section[i];
                }
                else
                {
                    x = _section[_wings + i] / BuildUp;
                    y = _section[_wings + i] % BuildUp;

                    if (_image[x, y] == 255)
                    {
                        loc = _section[i];
                    }
                }

                i++;
            }

            return loc;
        }

        int MakeStep(int startPoint)
        {
            int x = startPoint/BuildUp;
            int y = startPoint%BuildUp;

            double angle = _orientation.GetOrientation(x, y);

            x += Convert.ToInt32(_step * Math.Cos(angle));
            y += Convert.ToInt32(_step * Math.Sin(angle));

            return x*BuildUp + y;
        }

        bool CheckofStopCriteria()
        {
            return true;
        }

        void GoToLine()
        {
            
        }
    }
}
