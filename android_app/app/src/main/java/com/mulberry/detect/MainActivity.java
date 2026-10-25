package com.mulberry.detect;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.net.Uri;
import android.os.Bundle;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.ViewGroup;
import android.view.inputmethod.InputMethodManager;
import android.webkit.ValueCallback;
import android.webkit.WebChromeClient;
import android.webkit.WebResourceRequest;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

/**
 * 桑叶病虫害检测系统 —— 移动端(WebView 壳)。
 *
 * 界面与推理都由树莓派上的网页服务提供, 本应用负责用 WebView 承载,
 * 并额外解决两件浏览器里麻烦的事:
 *   1. 服务器地址可保存/修改(树莓派 IP 会随网络变化);
 *   2. 处理网页里的"选择文件"(上传图片检测), Android WebView 需要自己接管。
 */
public class MainActivity extends Activity {

    private static final String DEFAULT_URL = "http://172.20.10.4:7860";

    private WebView web;
    private LinearLayout setupView;
    private EditText input;
    private SharedPreferences sp;
    private ValueCallback<Uri[]> filePathCallback;   // 网页"选择文件"的回调
    private static final int REQ_FILE = 1001;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        sp = getSharedPreferences("cfg", MODE_PRIVATE);

        buildWebView();
        buildSetupView();

        String url = sp.getString("url", DEFAULT_URL);
        if (sp.getBoolean("configured", false)) {
            load(url);
            setContentView(web);
        } else {
            input.setText(url);
            setContentView(setupView);
        }
    }

    private void buildWebView() {
        web = new WebView(this);
        WebSettings s = web.getSettings();
        s.setJavaScriptEnabled(true);
        s.setDomStorageEnabled(true);
        s.setDatabaseEnabled(true);
        s.setLoadWithOverviewMode(true);
        s.setUseWideViewPort(true);
        s.setAllowFileAccess(true);
        s.setMediaPlaybackRequiresUserGesture(false);   // 允许自动播放/刷新画面
        s.setMixedContentMode(WebSettings.MIXED_CONTENT_ALWAYS_ALLOW);

        web.setWebViewClient(new WebViewClient() {
            @Override
            public boolean shouldOverrideUrlLoading(WebView v, WebResourceRequest req) {
                return false;   // 站内跳转都在本 WebView 内打开
            }
        });

        web.setWebChromeClient(new WebChromeClient() {
            @Override
            public boolean onShowFileChooser(WebView v, ValueCallback<Uri[]> cb,
                                             FileChooserParams params) {
                filePathCallback = cb;
                try {
                    startActivityForResult(params.createIntent(), REQ_FILE);
                    return true;
                } catch (Exception e) {
                    filePathCallback = null;
                    Toast.makeText(MainActivity.this, "无法打开文件选择器", Toast.LENGTH_SHORT).show();
                    return false;
                }
            }
        });
    }

    private void buildSetupView() {
        setupView = new LinearLayout(this);
        setupView.setOrientation(LinearLayout.VERTICAL);
        int pad = (int) (24 * getResources().getDisplayMetrics().density);
        setupView.setPadding(pad, pad, pad, pad);

        TextView title = new TextView(this);
        title.setText("桑叶病虫害检测系统");
        title.setTextSize(22);
        setupView.addView(title, wrap());

        TextView tip = new TextView(this);
        tip.setText("\n请输入树莓派上网页服务的地址(需与手机在同一局域网)。\n"
                + "在树莓派终端执行 ./webapp.sh status 可查看当前地址。\n");
        tip.setTextSize(14);
        setupView.addView(tip, wrap());

        input = new EditText(this);
        input.setHint(R.string.hint_server);
        input.setText(DEFAULT_URL);
        input.setSingleLine(true);
        setupView.addView(input, wrap());

        Button go = new Button(this);
        go.setText(R.string.btn_connect);
        go.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String url = input.getText().toString().trim();
                if (!url.startsWith("http://") && !url.startsWith("https://")) {
                    url = "http://" + url;
                }
                sp.edit().putString("url", url).putBoolean("configured", true).apply();
                hideKeyboard();
                load(url);
                setContentView(web);
            }
        });
        setupView.addView(go, wrap());
    }

    private LinearLayout.LayoutParams wrap() {
        return new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT);
    }

    private void load(String url) {
        web.loadUrl(url);
    }

    private void hideKeyboard() {
        InputMethodManager imm =
                (InputMethodManager) getSystemService(Context.INPUT_METHOD_SERVICE);
        if (imm != null) {
            imm.hideSoftInputFromWindow(input.getWindowToken(), 0);
        }
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        menu.add(0, 1, 0, R.string.set_server);
        menu.add(0, 2, 1, R.string.reload);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        if (item.getItemId() == 1) {
            input.setText(sp.getString("url", DEFAULT_URL));
            setContentView(setupView);
            return true;
        } else if (item.getItemId() == 2) {
            web.reload();
            return true;
        }
        return super.onOptionsItemSelected(item);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (requestCode == REQ_FILE) {
            if (filePathCallback != null) {
                Uri[] result = null;
                if (resultCode == RESULT_OK && data != null) {
                    if (data.getClipData() != null) {          // 多选
                        int n = data.getClipData().getItemCount();
                        result = new Uri[n];
                        for (int i = 0; i < n; i++) {
                            result[i] = data.getClipData().getItemAt(i).getUri();
                        }
                    } else if (data.getData() != null) {       // 单选
                        result = new Uri[]{data.getData()};
                    }
                }
                filePathCallback.onReceiveValue(result);
                filePathCallback = null;
            }
            return;
        }
        super.onActivityResult(requestCode, resultCode, data);
    }

    @Override
    public boolean onKeyDown(int keyCode, android.view.KeyEvent event) {
        if (keyCode == android.view.KeyEvent.KEYCODE_BACK
                && web != null && web.canGoBack()) {
            web.goBack();      // 返回键先在网页内后退
            return true;
        }
        return super.onKeyDown(keyCode, event);
    }
}
