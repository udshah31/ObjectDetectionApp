package com.younglords.objectdetectionapp

import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import com.younglords.objectdetectionapp.databinding.ActivityMainBinding
import com.younglords.objectdetectionapp.ml.SsdMobilenetV11Metadata1
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp

class MainActivity : AppCompatActivity() {


    private val paint = Paint()
    private var colors = listOf(
        Color.BLUE, Color.GREEN, Color.RED, Color.CYAN, Color.GRAY, Color.BLACK,
        Color.DKGRAY, Color.MAGENTA, Color.YELLOW, Color.RED
    )
    private lateinit var bitmap: Bitmap
    private lateinit var labels: List<String>
    private lateinit var model: SsdMobilenetV11Metadata1
    private val imageProcessor =
        ImageProcessor.Builder().add(ResizeOp(300, 300, ResizeOp.ResizeMethod.BILINEAR)).build()


    private val binding by lazy(LazyThreadSafetyMode.NONE) {
        ActivityMainBinding.inflate(layoutInflater)
    }


    private fun checkAndGrantPermissions() {
        if (checkSelfPermission(android.Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED) {
            requestPermissions(arrayOf(android.Manifest.permission.CAMERA), 100)
        } else {
            Toast.makeText(this, "Camera permission is granted", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray,
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 100) {
            if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "Camera permission is granted", Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(this, "Camera permission is denied", Toast.LENGTH_SHORT).show()
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(binding.root)

        labels = FileUtil.loadLabels(this, "labels.txt")
        model = SsdMobilenetV11Metadata1.newInstance(this)

        paint.color = Color.BLUE
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 3.0f


        binding.selectImage.setOnClickListener {
            selectImage()
        }

    }


    private fun selectImage() {

        val items = arrayOf<CharSequence>(
            "Take Photo", "Choose from Library", "Cancel"
        )

        val dialogBox = AlertDialog.Builder(this)
        dialogBox.setItems(items) { dialog, item ->
            when {
                items[item].contentEquals("Take Photo") -> {
                    openCamera()
                }
                items[item].contentEquals("Choose from Library") -> {
                    chooseImageFromGallery()
                }
                items[item].contentEquals("Cancel") -> {
                    dialog.dismiss()
                }
            }
        }
        dialogBox.setCancelable(true)
        dialogBox.show()
    }


    private fun chooseImageFromGallery() {
        val intent = Intent(Intent.ACTION_GET_CONTENT)
        intent.type = "image/*"
        intent.putExtra("crop", "true")
        intent.putExtra("scale", true)
        intent.putExtra("return-data", true)
        startActivityForResult(intent, 250)
    }

    private fun openCamera() {
        val camera = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        startActivityForResult(camera, 200)
    }


    @Deprecated("Deprecated in Java")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == 250) {
            val uri = data?.data
            bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)
            getPredictions()
        } else if (requestCode == 200 && resultCode == Activity.RESULT_OK) {
            bitmap = data?.extras?.get("data") as Bitmap
            binding.imageView.setImageBitmap(bitmap)
            getPredictions()
        }
    }


    private fun getPredictions() {
        var image = TensorImage.fromBitmap(bitmap)
        image = imageProcessor.process(image)

        val outputs = model.process(image)
        val locations = outputs.locationsAsTensorBuffer.floatArray
        val classes = outputs.classesAsTensorBuffer.floatArray
        val scores = outputs.scoresAsTensorBuffer.floatArray
        val numberOfDetections = outputs.numberOfDetectionsAsTensorBuffer.floatArray

        val mutable = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutable)
        val h = mutable.height
        val w = mutable.width

        paint.textSize = h / 15f
        paint.strokeWidth = h / 85f
        scores.forEachIndexed { index, fl ->
            if (fl > 0.5) {
                var x = index
                x *= 4
                paint.color = colors[index]
                paint.style = Paint.Style.STROKE
                canvas.drawRect(
                    RectF(
                        locations[x + 1] * w,
                        locations[x] * h,
                        locations[x + 3] * w,
                        locations[x + 2] * h
                    ), paint
                )
                paint.style = Paint.Style.FILL
                canvas.drawText(
                    labels[classes[index].toInt()] + " " + fl.toString(),
                    locations[x + 1] * w,
                    locations[x] * h,
                    paint
                )
            }
        }
        binding.imageView.setImageBitmap(mutable)
    }

    override fun onDestroy() {
        super.onDestroy()
        model.close()
    }
}