package org.bhaisoc.usi

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.os.Bundle
import android.webkit.JavascriptInterface
import android.webkit.WebChromeClient
import android.webkit.WebSettings
import android.webkit.WebView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.io.File

class MainActivity : AppCompatActivity() {

    private lateinit var webView: WebView
    private val CAMERA_PERMISSION_CODE = 1001

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        webView = WebView(this)
        setContentView(webView)

        configureWebView()
        requestCameraPermission()
    }

    @SuppressLint("SetJavaScriptEnabled")
    private fun configureWebView() {
        val settings: WebSettings = webView.settings
        settings.javaScriptEnabled = true
        settings.domStorageEnabled = true
        settings.allowFileAccess = true
        settings.allowContentAccess = true
        settings.mediaPlaybackRequiresUserGesture = false
        settings.cacheMode = WebSettings.LOAD_NO_CACHE

        webView.webChromeClient = WebChromeClient()
        webView.addJavascriptInterface(USiBridgeInterface(), "USiNative")

        // Load the pure dark USi on-chip UI from assets
        webView.loadUrl("file:///android_asset/index.html")
    }

    private fun requestCameraPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.CAMERA),
                CAMERA_PERMISSION_CODE
            )
        }
    }

    inner class USiBridgeInterface {
        @JavascriptInterface
        fun getModelStatus(): String {
            val modelDir = File(getExternalFilesDir(null), "models")
            val llamaFile = File(modelDir, "Llama-3.2-1B-Instruct-Q4_K_M.gguf")
            return if (llamaFile.exists()) {
                "{\"llama\": true, \"size_mb\": ${llamaFile.length() / (1024 * 1024)}}"
            } else {
                "{\"llama\": false, \"size_mb\": 0}"
            }
        }

        @JavascriptInterface
        fun getSoCInfo(): String {
            return "{\"soc\": \"Exynos 9611\", \"gpu\": \"ARM Mali-G72 MP3\", \"tier1_latency_ms\": 129.9, \"tier2_speed_tps\": 8.6}"
        }

        @JavascriptInterface
        fun showToast(message: String) {
            runOnUiThread {
                Toast.makeText(this@MainActivity, message, Toast.LENGTH_SHORT).show()
            }
        }
    }

    override fun onBackPressed() {
        if (webView.canGoBack()) {
            webView.goBack()
        } else {
            super.onBackPressed()
        }
    }
}
