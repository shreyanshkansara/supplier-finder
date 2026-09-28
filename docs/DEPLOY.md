# Deploying for free: Neon (database) + Render (app)

This takes about 20 minutes and costs nothing. No credit card is needed on either side at the time of writing; check the pricing pages, since free tiers change.

| Piece | Service | Free tier (September 2026) |
|---|---|---|
| Postgres + pgvector | [Neon](https://neon.com) | 0.5 GB storage and 100 compute-hours a month. Scales to zero after 5 idle minutes. This project uses about 5 MB |
| The app (Docker) | [Render](https://render.com) | 750 instance-hours a month. **Sleeps after 15 idle minutes**, and the first request after that takes about a minute to wake it |

Both free tiers sleep when idle. Before a demo or an interview, open the page a minute early so it's awake.

## 1. Put the code on GitHub

```bash
git init && git add . && git commit -m "Supplier Finder"
git branch -M main
git remote add origin https://github.com/<you>/supplier-finder.git
git push -u origin main
```

`.env` is in `.gitignore`, so your keys stay on your machine. Check with `git status` before pushing.

## 2. Create the database on Neon

1. Sign up at neon.com and create a project (any region; Singapore or Mumbai is closest to Jodhpur).
2. On the dashboard, click **Connect** and copy the connection string. It looks like
   `postgresql://user:password@ep-xxx.aws.neon.tech/neondb?sslmode=require&channel_binding=require`.
   Paste it exactly as it is. The app converts it for both of its Postgres drivers.

You don't need to create tables or enable pgvector yourself. The app's migrations do both when it starts.

## 3. Create the app on Render

1. Sign up at render.com with GitHub. Click **New → Blueprint** and pick your repository. Render reads `render.yaml`.
2. It asks for the values marked `sync: false`:
   - `DATABASE_URL`: the Neon string from step 2
   - `GEMINI_API_KEY`: your Google AI Studio key
   - `TYPESAFE_API_KEY`: your TypeSafe key
3. Click **Apply**. The first build takes a few minutes.
4. Open **Environment** on the service and copy the generated `API_KEY`. The web page asks for it once per browser.

## 4. First start

On the first boot, `SEED_ON_START=true` loads the 500 demo vendors in the background. Their Gemini search vectors ship with the repository (`data/embeddings/`), so this takes a few seconds and uses no Gemini quota. Watch **Logs** for `seeded 500 vendors`, or open `/health/ready` and wait for `"status": "ready"`. Later boots skip this (one quick count query).

Then open `https://<your-service>.onrender.com`, enter the API key, and try one of the example requests.

## Notes

- **Keys:** they live only in Render's environment settings and your local `.env`. Nothing in the repository contains a key.
- **Gemini free tier:** Google may use prompts sent on the free tier to improve its products. That's fine for this synthetic demo. Switch the key to the paid tier (the Google AI Pro subscription includes $10/month of Cloud credit) before any real business data goes through it.
- **Updating:** push to `main` and Render redeploys. Migrations run on every start and do nothing if the schema is current.
- **Other hosts:** the image is a plain Dockerfile that listens on `$PORT`. It runs anywhere that runs containers (Fly.io, Railway, Koyeb, a VPS) with any Postgres 16 that has pgvector.
