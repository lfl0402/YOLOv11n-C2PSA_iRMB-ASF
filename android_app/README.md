# Android 客户端（WebView）

这是一个很薄的 WebView 壳：界面与推理都由树莓派上的 `deploy/pi_app.py` 提供，App 只负责承载页面、保存服务器地址、处理网页里的“选择文件/拍照”上传。

## 文件

```text
android_app/
├── settings.gradle
├── build.gradle
├── gradle.properties
└── app/
    ├── build.gradle
    └── src/main/
        ├── AndroidManifest.xml
        ├── res/values/strings.xml
        ├── res/drawable/ic_launcher.png
        └── java/com/mulberry/detect/MainActivity.java
```

## 服务器地址

`MainActivity.java` 里：

```java
private static final String DEFAULT_URL = "http://raspberrypi.local:7860";
```

首次启动会显示地址设置页，也可以直接在网页里改。树莓派 IP 变化时，在 App 的设置页改地址即可，不需要重新打包。

## 构建

用 Android Studio 打开 `android_app/`，等待 Gradle Sync，然后 Run。

- `local.properties`（本机 Android SDK 路径）没有上传，Android Studio 会自动生成。
- 仓库里没有附带 APK：调试包会把局域网 IP 写死，换网络后不可用，建议按上面的方式自己构建。

## 权限

`AndroidManifest.xml` 需要网络权限；因为连的是 `http://` 局域网地址，WebView 已允许明文流量。若你的 Android 版本更严格，请在 `res/xml/network_security_config.xml` 中把树莓派 IP/域名加入白名单。
