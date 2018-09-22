namespace ImageMaskAnnotator
{
    partial class frmImgMaskAnnot
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.menuStrip1 = new System.Windows.Forms.MenuStrip();
            this.fileToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.openToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.saveToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripMenuItem1 = new System.Windows.Forms.ToolStripSeparator();
            this.nextToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.previousToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.lblImgPath = new System.Windows.Forms.Label();
            this.pbImgMask = new System.Windows.Forms.PictureBox();
            this.pbMask = new System.Windows.Forms.PictureBox();
            this.pbImg = new System.Windows.Forms.PictureBox();
            this.groupBox2 = new System.Windows.Forms.GroupBox();
            this.btnReset = new System.Windows.Forms.Button();
            this.rd11 = new System.Windows.Forms.RadioButton();
            this.rd3 = new System.Windows.Forms.RadioButton();
            this.pbPropMask = new System.Windows.Forms.PictureBox();
            this.pbBinMask = new System.Windows.Forms.PictureBox();
            this.pbOnlyPropMask = new System.Windows.Forms.PictureBox();
            this.menuStrip1.SuspendLayout();
            this.groupBox1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pbImgMask)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbMask)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbImg)).BeginInit();
            this.groupBox2.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pbPropMask)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbBinMask)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbOnlyPropMask)).BeginInit();
            this.SuspendLayout();
            // 
            // menuStrip1
            // 
            this.menuStrip1.ImageScalingSize = new System.Drawing.Size(24, 24);
            this.menuStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.fileToolStripMenuItem});
            this.menuStrip1.Location = new System.Drawing.Point(0, 0);
            this.menuStrip1.Name = "menuStrip1";
            this.menuStrip1.Size = new System.Drawing.Size(1174, 33);
            this.menuStrip1.TabIndex = 0;
            this.menuStrip1.Text = "menuStrip1";
            // 
            // fileToolStripMenuItem
            // 
            this.fileToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.openToolStripMenuItem,
            this.saveToolStripMenuItem,
            this.toolStripMenuItem1,
            this.nextToolStripMenuItem,
            this.previousToolStripMenuItem});
            this.fileToolStripMenuItem.Name = "fileToolStripMenuItem";
            this.fileToolStripMenuItem.Size = new System.Drawing.Size(50, 29);
            this.fileToolStripMenuItem.Text = "&File";
            // 
            // openToolStripMenuItem
            // 
            this.openToolStripMenuItem.Name = "openToolStripMenuItem";
            this.openToolStripMenuItem.ShortcutKeys = ((System.Windows.Forms.Keys)((System.Windows.Forms.Keys.Control | System.Windows.Forms.Keys.O)));
            this.openToolStripMenuItem.Size = new System.Drawing.Size(252, 30);
            this.openToolStripMenuItem.Text = "&Open...";
            this.openToolStripMenuItem.Click += new System.EventHandler(this.openToolStripMenuItem_Click);
            // 
            // saveToolStripMenuItem
            // 
            this.saveToolStripMenuItem.Name = "saveToolStripMenuItem";
            this.saveToolStripMenuItem.ShortcutKeys = ((System.Windows.Forms.Keys)((System.Windows.Forms.Keys.Control | System.Windows.Forms.Keys.S)));
            this.saveToolStripMenuItem.Size = new System.Drawing.Size(252, 30);
            this.saveToolStripMenuItem.Text = "&Save";
            this.saveToolStripMenuItem.Click += new System.EventHandler(this.saveToolStripMenuItem_Click);
            // 
            // toolStripMenuItem1
            // 
            this.toolStripMenuItem1.Name = "toolStripMenuItem1";
            this.toolStripMenuItem1.Size = new System.Drawing.Size(249, 6);
            // 
            // nextToolStripMenuItem
            // 
            this.nextToolStripMenuItem.Name = "nextToolStripMenuItem";
            this.nextToolStripMenuItem.ShortcutKeys = ((System.Windows.Forms.Keys)((System.Windows.Forms.Keys.Control | System.Windows.Forms.Keys.Right)));
            this.nextToolStripMenuItem.Size = new System.Drawing.Size(252, 30);
            this.nextToolStripMenuItem.Text = "&Next";
            this.nextToolStripMenuItem.Click += new System.EventHandler(this.nextToolStripMenuItem_Click);
            // 
            // previousToolStripMenuItem
            // 
            this.previousToolStripMenuItem.Name = "previousToolStripMenuItem";
            this.previousToolStripMenuItem.ShortcutKeys = ((System.Windows.Forms.Keys)((System.Windows.Forms.Keys.Control | System.Windows.Forms.Keys.Left)));
            this.previousToolStripMenuItem.Size = new System.Drawing.Size(252, 30);
            this.previousToolStripMenuItem.Text = "&Previous";
            this.previousToolStripMenuItem.Click += new System.EventHandler(this.previousToolStripMenuItem_Click);
            // 
            // groupBox1
            // 
            this.groupBox1.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.groupBox1.Controls.Add(this.lblImgPath);
            this.groupBox1.Controls.Add(this.pbImgMask);
            this.groupBox1.Controls.Add(this.pbBinMask);
            this.groupBox1.Controls.Add(this.pbMask);
            this.groupBox1.Controls.Add(this.pbImg);
            this.groupBox1.Location = new System.Drawing.Point(12, 36);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(1150, 276);
            this.groupBox1.TabIndex = 1;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Image / mask / image+mask";
            // 
            // lblImgPath
            // 
            this.lblImgPath.AutoSize = true;
            this.lblImgPath.Font = new System.Drawing.Font("Microsoft Sans Serif", 14F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblImgPath.Location = new System.Drawing.Point(23, 26);
            this.lblImgPath.Name = "lblImgPath";
            this.lblImgPath.Size = new System.Drawing.Size(120, 32);
            this.lblImgPath.TabIndex = 1;
            this.lblImgPath.Text = "ImgPath";
            // 
            // pbImgMask
            // 
            this.pbImgMask.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.pbImgMask.Location = new System.Drawing.Point(761, 70);
            this.pbImgMask.Name = "pbImgMask";
            this.pbImgMask.Size = new System.Drawing.Size(227, 191);
            this.pbImgMask.TabIndex = 0;
            this.pbImgMask.TabStop = false;
            // 
            // pbMask
            // 
            this.pbMask.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.pbMask.Location = new System.Drawing.Point(256, 70);
            this.pbMask.Name = "pbMask";
            this.pbMask.Size = new System.Drawing.Size(227, 191);
            this.pbMask.TabIndex = 0;
            this.pbMask.TabStop = false;
            // 
            // pbImg
            // 
            this.pbImg.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.pbImg.Location = new System.Drawing.Point(23, 70);
            this.pbImg.Name = "pbImg";
            this.pbImg.Size = new System.Drawing.Size(227, 191);
            this.pbImg.TabIndex = 0;
            this.pbImg.TabStop = false;
            // 
            // groupBox2
            // 
            this.groupBox2.Controls.Add(this.pbOnlyPropMask);
            this.groupBox2.Controls.Add(this.btnReset);
            this.groupBox2.Controls.Add(this.rd11);
            this.groupBox2.Controls.Add(this.rd3);
            this.groupBox2.Controls.Add(this.pbPropMask);
            this.groupBox2.Location = new System.Drawing.Point(12, 318);
            this.groupBox2.Name = "groupBox2";
            this.groupBox2.Size = new System.Drawing.Size(689, 238);
            this.groupBox2.TabIndex = 2;
            this.groupBox2.TabStop = false;
            this.groupBox2.Text = "Proposed Mask";
            // 
            // btnReset
            // 
            this.btnReset.Location = new System.Drawing.Point(275, 103);
            this.btnReset.Name = "btnReset";
            this.btnReset.Size = new System.Drawing.Size(75, 52);
            this.btnReset.TabIndex = 3;
            this.btnReset.Text = "Reset";
            this.btnReset.UseVisualStyleBackColor = true;
            this.btnReset.Click += new System.EventHandler(this.btnReset_Click);
            // 
            // rd11
            // 
            this.rd11.AutoSize = true;
            this.rd11.Location = new System.Drawing.Point(275, 62);
            this.rd11.Name = "rd11";
            this.rd11.Size = new System.Drawing.Size(59, 24);
            this.rd11.TabIndex = 2;
            this.rd11.Text = "7x7";
            this.rd11.UseVisualStyleBackColor = true;
            this.rd11.CheckedChanged += new System.EventHandler(this.rd11_CheckedChanged);
            // 
            // rd3
            // 
            this.rd3.AutoSize = true;
            this.rd3.Checked = true;
            this.rd3.Location = new System.Drawing.Point(275, 32);
            this.rd3.Name = "rd3";
            this.rd3.Size = new System.Drawing.Size(59, 24);
            this.rd3.TabIndex = 2;
            this.rd3.TabStop = true;
            this.rd3.Text = "3x3";
            this.rd3.UseVisualStyleBackColor = true;
            this.rd3.CheckedChanged += new System.EventHandler(this.rd3_CheckedChanged);
            // 
            // pbPropMask
            // 
            this.pbPropMask.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.pbPropMask.Location = new System.Drawing.Point(19, 32);
            this.pbPropMask.Name = "pbPropMask";
            this.pbPropMask.Size = new System.Drawing.Size(227, 191);
            this.pbPropMask.SizeMode = System.Windows.Forms.PictureBoxSizeMode.AutoSize;
            this.pbPropMask.TabIndex = 1;
            this.pbPropMask.TabStop = false;
            this.pbPropMask.Paint += new System.Windows.Forms.PaintEventHandler(this.pbPropMask_Paint);
            this.pbPropMask.MouseDown += new System.Windows.Forms.MouseEventHandler(this.pbPropMask_MouseDown);
            this.pbPropMask.MouseMove += new System.Windows.Forms.MouseEventHandler(this.pbPropMask_MouseMove);
            this.pbPropMask.MouseUp += new System.Windows.Forms.MouseEventHandler(this.pbPropMask_MouseUp);
            // 
            // pbBinMask
            // 
            this.pbBinMask.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.pbBinMask.Location = new System.Drawing.Point(489, 70);
            this.pbBinMask.Name = "pbBinMask";
            this.pbBinMask.Size = new System.Drawing.Size(227, 191);
            this.pbBinMask.TabIndex = 0;
            this.pbBinMask.TabStop = false;
            // 
            // pbOnlyPropMask
            // 
            this.pbOnlyPropMask.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.pbOnlyPropMask.Location = new System.Drawing.Point(418, 32);
            this.pbOnlyPropMask.Name = "pbOnlyPropMask";
            this.pbOnlyPropMask.Size = new System.Drawing.Size(227, 191);
            this.pbOnlyPropMask.SizeMode = System.Windows.Forms.PictureBoxSizeMode.AutoSize;
            this.pbOnlyPropMask.TabIndex = 4;
            this.pbOnlyPropMask.TabStop = false;
            // 
            // frmImgMaskAnnot
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(9F, 20F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1174, 568);
            this.Controls.Add(this.groupBox2);
            this.Controls.Add(this.groupBox1);
            this.Controls.Add(this.menuStrip1);
            this.MainMenuStrip = this.menuStrip1;
            this.Name = "frmImgMaskAnnot";
            this.Text = "ImageMaskAnnotator";
            this.WindowState = System.Windows.Forms.FormWindowState.Maximized;
            this.Load += new System.EventHandler(this.frmImgMaskAnnot_Load);
            this.menuStrip1.ResumeLayout(false);
            this.menuStrip1.PerformLayout();
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pbImgMask)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbMask)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbImg)).EndInit();
            this.groupBox2.ResumeLayout(false);
            this.groupBox2.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pbPropMask)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbBinMask)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbOnlyPropMask)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.MenuStrip menuStrip1;
        private System.Windows.Forms.ToolStripMenuItem fileToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem openToolStripMenuItem;
        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.PictureBox pbImg;
        private System.Windows.Forms.PictureBox pbImgMask;
        private System.Windows.Forms.PictureBox pbMask;
        private System.Windows.Forms.ToolStripSeparator toolStripMenuItem1;
        private System.Windows.Forms.ToolStripMenuItem nextToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem previousToolStripMenuItem;
        private System.Windows.Forms.Label lblImgPath;
        private System.Windows.Forms.GroupBox groupBox2;
        private System.Windows.Forms.PictureBox pbPropMask;
        private System.Windows.Forms.RadioButton rd11;
        private System.Windows.Forms.RadioButton rd3;
        private System.Windows.Forms.Button btnReset;
        private System.Windows.Forms.ToolStripMenuItem saveToolStripMenuItem;
        private System.Windows.Forms.PictureBox pbBinMask;
        private System.Windows.Forms.PictureBox pbOnlyPropMask;
    }
}

