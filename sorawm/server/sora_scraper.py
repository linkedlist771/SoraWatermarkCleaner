import os
import platform
import asyncio
from pathlib import Path
from typing import Optional
from playwright.async_api import async_playwright, BrowserContext
from loguru import logger

class SoraScraper:
    def __init__(self):
        self.browser_context: Optional[BrowserContext] = None
        self.playwright = None

    def get_chrome_path(self) -> Optional[str]:
        """Detects Chrome executable path on Windows."""
        if platform.system() != "Windows":
            return None

        # Common paths for Chrome on Windows
        paths = [
            os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
            os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
            os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe"),
        ]

        for path in paths:
            if os.path.exists(path):
                return path
        return None

    def get_user_data_dir(self) -> str:
        """Detects Chrome user data directory on Windows."""
        if platform.system() == "Windows":
            return os.path.expandvars(r"%LocalAppData%\Google\Chrome\User Data")
        # Fallback for other OS or if custom logic needed later
        return str(Path.home() / ".config" / "google-chrome")

    async def launch_browser(self, headless: bool = False):
        """Launches Chrome with the user's persistent profile."""
        if self.browser_context:
            return self.browser_context

        self.playwright = await async_playwright().start()

        executable_path = self.get_chrome_path()
        user_data_dir = self.get_user_data_dir()

        logger.info(f"Launching Chrome from: {executable_path}")
        logger.info(f"Using User Data Dir: {user_data_dir}")

        try:
            # We use launch_persistent_context to use the actual user profile
            self.browser_context = await self.playwright.chromium.launch_persistent_context(
                user_data_dir=user_data_dir,
                executable_path=executable_path,
                headless=headless,
                accept_downloads=True,
                args=["--no-sandbox", "--disable-blink-features=AutomationControlled"] # Helper args to avoid detection
            )
            return self.browser_context
        except Exception as e:
            logger.error(f"Failed to launch Chrome: {e}")
            # Fallback to standard launch if persistent context fails (e.g. browser already open)
            logger.warning("Attempting fallback launch (clean profile)...")
            browser = await self.playwright.chromium.launch(
                executable_path=executable_path,
                headless=headless
            )
            self.browser_context = await browser.new_context()
            return self.browser_context

    async def login(self):
        """Opens Sora login page for the user to interact with."""
        context = await self.launch_browser(headless=False) # Must be visible for login
        page = await context.new_page()
        await page.goto("https://sora.chatgpt.com/")
        logger.info("Opened Sora login page.")
        # We don't close it, we leave it open for the user

    async def download_video(self, url: str, output_path: Path) -> bool:
        """Navigates to the URL and attempts to download the video."""
        # For downloading, we can try headless, but sometimes headful is safer for cookies
        context = await self.launch_browser(headless=False)
        page = await context.new_page()

        try:
            logger.info(f"Navigating to {url}")
            await page.goto(url, wait_until="networkidle")

            # Wait for video element
            try:
                # Try to find the video tag
                video_element = await page.wait_for_selector("video", timeout=10000)
                if not video_element:
                    raise Exception("No video element found")

                src = await video_element.get_attribute("src")
                if not src:
                    # Sometimes src is blob, which is harder.
                    # Let's hope for a direct link or we might need to intercept network requests.
                     logger.info("Video src is empty, checking network requests...")

                logger.info(f"Found video URL: {src}")

                # If src is a blob, we might need to handle it differently.
                # For now, let's assume it's a URL we can download.

                if src:
                    # Download the content
                    response = await page.request.get(src)
                    if response.status == 200:
                        body = await response.body()
                        with open(output_path, "wb") as f:
                            f.write(body)
                        logger.info(f"Downloaded video to {output_path}")
                        return True
                    else:
                        logger.error(f"Failed to download video: Status {response.status}")

            except Exception as e:
                logger.error(f"Error finding video: {e}")

        except Exception as e:
            logger.error(f"Error during navigation: {e}")
        finally:
            await page.close()
            # We keep the context open for future requests or let it close when app exits

        return False

    async def close(self):
        if self.browser_context:
            await self.browser_context.close()
        if self.playwright:
            await self.playwright.stop()

sora_scraper = SoraScraper()
