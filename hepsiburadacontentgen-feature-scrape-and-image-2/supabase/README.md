# Supabase Migrations

This directory contains database migrations for Supabase, including functions, triggers, and other database objects.

## Directory Structure

- `schema.sql`: Base schema definition
- `migrations/`: Individual migration files, prefixed with timestamps

## Credit Decrease Trigger

The key functionality implemented is a trigger that automatically decreases the credit count when a new generation is inserted into the `generations` table.

### How it works

1. When a new row is inserted into the `generations` table
2. The `after_generation_insert` trigger fires
3. The `decrease_credit()` function is called
4. The function updates the `credit` column of the newly inserted row, decreasing it by 1
5. The function returns the updated credit value

## Deploying Migrations

### Automatic Deployment

Migrations are automatically deployed as part of the `deploy.sh` script which calls `deploy_supabase_migrations.sh`.

### Manual Deployment

To manually deploy the migrations:

```bash
# Make sure you have the required environment variables
export SUPABASE_URL=your_supabase_url
export SUPABASE_KEY=your_supabase_key

# Run the deployment script
./deploy_supabase_migrations.sh
```

## Troubleshooting

If the trigger is not working:

1. Verify that the function and trigger exist in the database:
   ```sql
   SELECT * FROM pg_proc WHERE proname = 'decrease_credit';
   SELECT * FROM pg_trigger WHERE tgname = 'after_generation_insert';
   ```

2. Check that the function has the correct permissions:
   ```sql
   -- Re-create function if needed with correct permissions
   CREATE OR REPLACE FUNCTION public.decrease_credit()
   RETURNS TRIGGER AS $$
   DECLARE
       current_credit integer;
   BEGIN
       UPDATE public.generations
       SET credit = credit - 1
       WHERE id = NEW.id
       RETURNING credit INTO current_credit;
       
       RAISE NOTICE 'Credit decreased for generation ID: %. New credit: %', NEW.id, current_credit;
       
       RETURN NEW;
   END;
   $$ LANGUAGE plpgsql SECURITY DEFINER;
   ```

3. Manually verify the trigger operation with a test insert:
   ```sql
   -- Insert a test record and check if credit decreases
   INSERT INTO public.generations (/* required fields */) VALUES (/* values */);
   SELECT id, credit FROM public.generations ORDER BY id DESC LIMIT 1;
   ``` 