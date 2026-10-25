# Android 客户端（WebView）

这是一个很薄的 WebView 壳：界面与推理都由树莓派上的 `deploy/pi_app.py` 提供，App 只负责承载页面、保存服务器地址、处理网页里的“选择文件/拍照”上传。APK 约 18 KB，推理不在手机上跑。

## 下载 / 安装

- **GitHub Release**：https://github.com/lfl0402/YOLOv11n-C2PSA_iRMB-ASF/releases/tag/v1.0.0
- 直链：https://github.com/lfl0402/YOLOv11n-C2PSA_iRMB-ASF/releases/download/v1.0.0/mulberry-detect.apk
- 本地构建产物：`android_app/release/mulberry-detect.apk`

安装：

1. 手机与树莓派连接同一个 Wi-Fi / 热点。
2. 安装 APK；这是 Debug 包，Android 会提示“未知来源”，允许即可。
3. 首次启动输入 `http://172.20.10.4:7860`，点“连接”。
4. 如果树莓派 IP 变了，在 App 的设置页改地址，不需要重新打包。

> 当前默认地址是 `172.20.10.4:7860`。树莓派上的 `pi_app.py` 需要先启动，手机才能打开页面。

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
private static final String DEFAULT_URL = "http://172.20.10.4:7860";
```

首次启动会显示地址设置页，也可以直接在网页里改。树莓派 IP 变化时，在 App 的设置页改地址即可。

## 构建

用 Android Studio 打开 `android_app/`，等待 Gradle Sync，然后 Run。

也可以命令行构建（需要 JDK 17、Android SDK 34、Gradle 8.7）：

```bash
cd android_app
# local.properties 写 your sdk.dir，例如：
# sdk.dir=D\:\\Android\\sdk
gradle assembleDebug
```

产物在 `app/build/outputs/apk/debug/app-debug.apk`。

- `local.properties` 是每台机器的 SDK 路径，不提交。
- `app/build/`、`.gradle/` 是构建缓存，不提交。
- Release 里的 APK 是 Debug 签名，仅用于课程/面试演示；正式分发请用自己的 keystore 签名。

## 权限

`AndroidManifest.xml` 需要网络和相机权限；因为连的是 `http://` 局域网地址，WebView 已允许明文流量。若你的 Android 版本更严格，请在 `res/xml/network_security_config.xml` 中把树莓派 IP/域名加入白名单。
