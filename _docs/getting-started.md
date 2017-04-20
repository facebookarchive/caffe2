---
docid: getting-started
title: Install
layout: docs-getting-started
permalink: /docs/getting-started.html
---
{% capture mac %}{% include_relative getting-started/mac.md %}{% endcapture %}
{% capture ubuntu %}{% include_relative getting-started/ubuntu.md %}{% endcapture %}
{% capture centos %}{% include_relative getting-started/centos.md %}{% endcapture %}
{% capture windows %}{% include_relative getting-started/windows.md %}{% endcapture %}
{% capture ios %}{% include_relative getting-started/ios.md %}{% endcapture %}
{% capture android %}{% include_relative getting-started/android.md %}{% endcapture %}
{% capture docker %}{% include_relative getting-started/docker.md %}{% endcapture %}
{% capture raspbian %}{% include_relative getting-started/raspbian.md %}{% endcapture %}
{% capture tegra %}{% include_relative getting-started/tegra.md %}{% endcapture %}

{{mac | markdownify }}
{{ubuntu | markdownify }}
{{centos | markdownify }}
{{windows | markdownify }}
{{ios | markdownify }}
{{android | markdownify }}
{{docker | markdownify }}
{{raspbian | markdownify }}
{{tegra | markdownify }}
