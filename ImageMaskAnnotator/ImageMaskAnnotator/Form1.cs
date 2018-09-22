using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

using System.IO;

namespace ImageMaskAnnotator
{
    public partial class frmImgMaskAnnot : Form
    {
        public frmImgMaskAnnot()
        {
            InitializeComponent();
        }

        /// <summary>Proposed mask</summary>
        float[,,] propMask = null;
        /// <summary>Raw image pixel data</summary>
        float[,,] pixels = null;

        #region Open Files
        Bitmap bmpImg, bmpMask;


        string imgFolder;
        int imgId;
        FileInfo[] allFiles;
        FileInfo[] procFiles;
        private void openFile(string filename)
        {
            string fname = Path.GetFileName(filename);
            string maskPath = Directory.GetParent(filename).Parent.FullName + "\\masks\\" + fname;

            DirectoryInfo basePath = Directory.GetParent(filename).Parent;
            procFiles = basePath.GetFiles("img*.png");

            bmpImg = new Bitmap(filename);

            pbImg.Width = bmpImg.Width;
            pbImg.Height = bmpImg.Height;
            pbImg.Image = bmpImg;
            pixels = ImgMaskCreator.GetBmpIntens(bmpImg);

            if (File.Exists(maskPath))
            {
                bmpMask = new Bitmap(maskPath);
                pbMask.Image = bmpMask;
                pbMask.Width = bmpMask.Width;
                pbMask.Height = bmpMask.Height;

                float[,,] pixMask = ImgMaskCreator.GetBmpIntens(bmpMask);
                propMask = ImgMaskCreator.GetBmpIntens(bmpMask);

                Bitmap bmpBinMask = ImgMaskCreator.GetBitmapFromIntens(pixMask, true);
                pbBinMask.Image = bmpBinMask;
                pbBinMask.Width = bmpBinMask.Width;
                pbBinMask.Height = bmpBinMask.Height;

                Bitmap bmpImgMask = ImgMaskCreator.GetMaskedBitmap(pixels, pixMask);
                pbImgMask.Image = bmpImgMask;
                pbImgMask.Width = bmpImgMask.Width;
                pbImgMask.Height = bmpImgMask.Height;
            }
            else
            {
                propMask = new float[bmpImg.Width, bmpImg.Height,3];
            }

            redrawPropMask();
            //pbPropMask.Image = ImgMaskCreator.GetMaskedBitmap(pixels, propMask);
            //pbPropMask.Width = bmpImg.Width;
            //pbPropMask.Height = bmpImg.Height;

            lblImgPath.Text = filename;
        }

        private void nextToolStripMenuItem_Click(object sender, EventArgs e)
        {
            imgId++;

            while (imgId < allFiles.Length && procFiles.Where(p => p.Name == "img" + allFiles[imgId].Name).Count() > 0)
            {
                imgId++;
            }

            if (imgId >= allFiles.Length) imgId = allFiles.Length - 1;
            if (allFiles != null) openFile(allFiles[imgId].FullName);
        }

        private void previousToolStripMenuItem_Click(object sender, EventArgs e)
        {
            imgId--;

            while (imgId>=0 && procFiles.Where(p => p.Name == "img" + allFiles[imgId].Name).Count() > 0)
            {
                imgId--;
            }

            if (imgId < 0) imgId = 0;
            if (allFiles != null) openFile(allFiles[imgId].FullName);
        }

        private void frmImgMaskAnnot_Load(object sender, EventArgs e)
        {

        }

        private void openToolStripMenuItem_Click(object sender, EventArgs e)
        {
            OpenFileDialog ofd = new OpenFileDialog();
            ofd.Filter = "Images|*.png;*.bmp;*.jpg;*.jpeg;*.tiff";
            if (ofd.ShowDialog() == DialogResult.OK )
            {
                imgFolder = Path.GetDirectoryName(ofd.FileName);
                DirectoryInfo di = new DirectoryInfo(imgFolder);
                allFiles = di.GetFiles();
                string fname = Path.GetFileName(ofd.FileName);

                for (int i = 0; i < allFiles.Length; i++)
                {
                    if (allFiles[i].Name == fname)
                    {
                        imgId = i;
                        break;
                    }
                }

                openFile(allFiles[imgId].FullName);

            }
            clicked = false;
        }
        #endregion

        #region Edit mask

        int brushSize = 11;
        private void rd3_CheckedChanged(object sender, EventArgs e)
        {
            brushSize = 3;
        }
        private void rd11_CheckedChanged(object sender, EventArgs e)
        {
            brushSize = 7;
        }
        private void btnReset_Click(object sender, EventArgs e)
        {
            for (int x = 0; x < propMask.GetLength(0); x++)
                for (int y = 0; y < propMask.GetLength(1); y++)
                    for (int z = 0; z < propMask.GetLength(2); z++)
                        propMask[x, y, z] = 0;

            redrawPropMask();
        }

        bool clicked = false;
        bool clickRight = false;
        private void pbPropMask_MouseDown(object sender, MouseEventArgs e)
        {
            if (e.Button == MouseButtons.Left) clicked = true;
            else clickRight = true;
        }
        private void pbPropMask_MouseUp(object sender, MouseEventArgs e)
        {
            clicked = false;
            clickRight = false;
        }

        PointF mousept = new PointF();
        private void pbPropMask_MouseMove(object sender, MouseEventArgs e)
        {
            if (propMask == null) return;

            mousept = new PointF(e.X, e.Y);
            if (clicked || clickRight)
            {
                for (int x = -brushSize; x <= brushSize; x++)
                    for (int y = -brushSize; y <= brushSize; y++)
                    {
                        int px = e.X + x;
                        int py = e.Y + y;
                        if (px >= 0 && py >= 0 && px < propMask.GetLength(0) && py < propMask.GetLength(1))
                        {
                            int desValue = clicked ? 255 : 0;
                            for (int z = 0; z < 3; z++) propMask[px, py, z] = desValue;
                        }
                    }
                redrawPropMask();
            }
            pbPropMask.Invalidate();
        }

        private void pbPropMask_Paint(object sender, PaintEventArgs e)
        {
            e.Graphics.DrawRectangle(Pens.Yellow, mousept.X - brushSize, mousept.Y - brushSize, 2 * brushSize, 2 * brushSize);

        }

        private void saveToolStripMenuItem_Click(object sender, EventArgs e)
        {
            string fname = Path.GetFileName(lblImgPath.Text);
            string imgFolder = Directory.GetParent(lblImgPath.Text).Parent.FullName+"\\";
            
            bmpImg.Save(imgFolder + "img" + fname, System.Drawing.Imaging.ImageFormat.Png);

            Bitmap maskbmp = ImgMaskCreator.GetBitmapFromIntens(propMask, true);
            maskbmp.Save(imgFolder + "msk" + fname, System.Drawing.Imaging.ImageFormat.Png);
        }


        /// <summary>Redraw proposed mask</summary>
        private void redrawPropMask()
        {
            pbPropMask.Image = ImgMaskCreator.GetMaskedBitmap(pixels, propMask);
            pbOnlyPropMask.Image = ImgMaskCreator.GetBitmapFromIntens(propMask, true);
            pbPropMask.Invalidate();
        }



        #endregion
    }
}
