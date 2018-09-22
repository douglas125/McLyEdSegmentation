using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace ImageMaskAnnotator
{
    class ImgMaskCreator
    {
        #region Bitmap pixel read/write

        public static Bitmap GetMaskedBitmap(float[,,] vals, float[,,] mask)
        {
            Bitmap processedBitmap = new Bitmap(vals.GetLength(0), vals.GetLength(1));

            int W = processedBitmap.Width;
            BitmapData bitmapData = processedBitmap.LockBits(new Rectangle(0, 0, processedBitmap.Width, processedBitmap.Height), ImageLockMode.ReadWrite, processedBitmap.PixelFormat);

            int bytesPerPixel = Bitmap.GetPixelFormatSize(processedBitmap.PixelFormat) / 8;
            int byteCount = bitmapData.Stride * processedBitmap.Height;
            byte[] pixels = new byte[byteCount];
            IntPtr ptrFirstPixel = bitmapData.Scan0;
            Marshal.Copy(ptrFirstPixel, pixels, 0, pixels.Length);
            int heightInPixels = bitmapData.Height;
            int widthInBytes = bitmapData.Width * bytesPerPixel;

            int xx, yy;
            yy = 0;

            float f = 0.9f;
            for (int y = 0; y < heightInPixels; y++)
            {
                int currentLine = y * bitmapData.Stride;
                xx = 0;
                for (int x = 0; x < widthInBytes; x = x + bytesPerPixel)
                {
                    byte r = (byte)(vals[xx, yy, 0]*f);
                    byte g = (byte)(vals[xx, yy, 1] * f + (1.0f - f) * mask[xx, yy, 1]);
                    byte b = (byte)(vals[xx, yy, 2]*f);

                    pixels[currentLine + x] = b;
                    pixels[currentLine + x + 1] = g;
                    pixels[currentLine + x + 2] = r;
                    pixels[currentLine + x + 3] = 255;

                    xx++;
                }
                yy++;
            }

            // copy modified bytes back
            Marshal.Copy(pixels, 0, ptrFirstPixel, pixels.Length);
            processedBitmap.UnlockBits(bitmapData);

            return processedBitmap;
        }


        public static Bitmap GetBitmapFromIntens(float[,,] vals, bool binarize = false)
        {
            Bitmap processedBitmap = new Bitmap(vals.GetLength(0), vals.GetLength(1));

            int W = processedBitmap.Width;
            BitmapData bitmapData = processedBitmap.LockBits(new Rectangle(0, 0, processedBitmap.Width, processedBitmap.Height), ImageLockMode.ReadWrite, processedBitmap.PixelFormat);

            int bytesPerPixel = Bitmap.GetPixelFormatSize(processedBitmap.PixelFormat) / 8;
            int byteCount = bitmapData.Stride * processedBitmap.Height;
            byte[] pixels = new byte[byteCount];
            IntPtr ptrFirstPixel = bitmapData.Scan0;
            Marshal.Copy(ptrFirstPixel, pixels, 0, pixels.Length);
            int heightInPixels = bitmapData.Height;
            int widthInBytes = bitmapData.Width * bytesPerPixel;

            int xx, yy;
            yy = 0;

            for (int y = 0; y < heightInPixels; y++)
            {
                int currentLine = y * bitmapData.Stride;
                xx = 0;
                for (int x = 0; x < widthInBytes; x = x + bytesPerPixel)
                {
                    byte r = (byte)vals[xx, yy, 0];
                    byte g = (byte)vals[xx, yy, 1];
                    byte b = (byte)vals[xx, yy, 2];

                    if (binarize)
                    {
                        if (r > 127) r = 255; else r = 0;
                        if (g > 127) g = 255; else g = 0;
                        if (b > 127) b = 255; else b = 0;
                    }

                    pixels[currentLine + x] = b;
                    pixels[currentLine + x + 1] = g;
                    pixels[currentLine + x + 2] = r;
                    pixels[currentLine + x + 3] = 255;

                    xx++;
                }
                yy++;
            }

            // copy modified bytes back
            Marshal.Copy(pixels, 0, ptrFirstPixel, pixels.Length);
            processedBitmap.UnlockBits(bitmapData);

            return processedBitmap;
        }

        public static float[,,] GetBmpIntens(Bitmap processedBitmap)
        {
            int W = processedBitmap.Width;
            BitmapData bitmapData = processedBitmap.LockBits(new Rectangle(0, 0, processedBitmap.Width, processedBitmap.Height), ImageLockMode.ReadWrite, processedBitmap.PixelFormat);

            float[,,] vals = new float[W, processedBitmap.Height,3];


            int bytesPerPixel = Bitmap.GetPixelFormatSize(processedBitmap.PixelFormat) / 8;
            int byteCount = bitmapData.Stride * processedBitmap.Height;
            byte[] pixels = new byte[byteCount];
            IntPtr ptrFirstPixel = bitmapData.Scan0;
            Marshal.Copy(ptrFirstPixel, pixels, 0, pixels.Length);
            int heightInPixels = bitmapData.Height;
            int widthInBytes = bitmapData.Width * bytesPerPixel;

            int xx, yy;
            yy = 0;

            for (int y = 0; y < heightInPixels; y++)
            {
                int currentLine = y * bitmapData.Stride;
                xx = 0;
                for (int x = 0; x < widthInBytes; x = x + bytesPerPixel)
                {
                    int oldBlue = pixels[currentLine + x];
                    int oldGreen = pixels[currentLine + x + 1];
                    int oldRed = pixels[currentLine + x + 2];

                    //float[] rgb = FcnGetColor(vals[xx + W * yy], min, max);


                    // calculate new pixel value
                    vals[xx, yy, 0] = oldRed; //limit range to [0,1];
                    vals[xx, yy, 1] = oldGreen; //limit range to [0,1];
                    vals[xx, yy, 2] = oldBlue; //limit range to [0,1];

                    xx++;
                }
                yy++;
            }

            // copy modified bytes back
            Marshal.Copy(pixels, 0, ptrFirstPixel, pixels.Length);
            processedBitmap.UnlockBits(bitmapData);

            return vals;
        }
        #endregion
    }
}
