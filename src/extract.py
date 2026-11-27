"""Copy the tables the models need from MySQL into data/raw/*.parquet (read-only session).
Set DATABASE_URL (mysql://user:pass@host:3306/db); use a read replica or the weekly copy, not the live primary."""
import os, sys, urllib.parse, pymysql, pandas as pd
u = urllib.parse.urlparse(os.environ["DATABASE_URL"])
c = pymysql.connect(host=u.hostname, user=u.username, password=urllib.parse.unquote(u.password), database=u.path.lstrip("/"), ssl={"ssl":{}}, read_timeout=600)
c.cursor().execute("SET SESSION TRANSACTION READ ONLY")
Q = {
"deals": "SELECT * FROM deals",
"deals_list": "SELECT * FROM deals_list",
"groups_list": "SELECT * FROM groups_list",
"sub_stages": "SELECT id, company_id, list_id, name FROM deal_sub_stages",
"customers": "SELECT id,name,phone,phone_2,address,city,state,postal_code,notes,created_date,company_id,company_name,referral_source,referral_source_raw,source,sales_rep,remodal_type,project_size,contact_time,remove_and_dispose,improve_offer,sink,when_start,details,compaign_name,adset_name,ad_name,backsplash,kitchen_stove,your_message,attached_file,invalid_lead,assigned_date,deleted_at,parent_id,created_by,email_id,form_name,customer_temperature FROM customers",
"customers_emails": "SELECT * FROM customers_emails",
"users": "SELECT id,name,company_id,created_date,is_employee,is_admin,is_deleted,lead_automation FROM users",
"stage_history": "SELECT * FROM deal_stage_history",
"activities": "SELECT * FROM deal_activities",
"notes": "SELECT * FROM deal_notes",
"cust_history": "SELECT * FROM customers_history",
"quotes": "SELECT id,customer_id,quote_name,quote_type,created_date,sales_rep,company_id,deleted_at,deal_id,status,salesperson_id,created_by,price_list_id,expiration_date,notes,tax_pct,discount_pct,address,current_step,current_revision,signed_at,customer_accepted_option_id,updated_at,discount_amount,qbo_invoice_id,sale_new_deal_id,sale_prior_deal_id,LENGTH(draft_drawing_json) draft_len FROM quotes",
"quote_revisions": "SELECT id,quote_id,revision_number,drawing_json,options_json,totals_json,notes,created_by,created_at,has_cutout FROM quote_revisions",
"quote_emails": "SELECT id,quote_id,revision_number,sent_by,sent_at,view_count,first_viewed_at,last_viewed_at,deactivated_at,form FROM quote_emails",
"quote_email_views": "SELECT * FROM quote_email_views",
"sales": "SELECT id,customer_id,deal_id,seller_id,price,company_id,sale_date,status,cancelled_date,installed_date,square_feet,paid_date FROM sales",
"sales_orders": "SELECT * FROM sales_orders",
"customer_payments": "SELECT id,company_id,customer_id,deal_id,sale_id,method,status,amount_cents,created_at,paid_at FROM customer_payments",
"events": "SELECT id,title,description,location,start_date,end_date,all_day,status,created_user_id,assigned_user_id,sale_id,customer_id,quote_id,calendar_id,created_date,deleted_date,project_id FROM events",
"emails": "SELECT id,subject,body,sent_at,deleted_at,thread_id,sender_user_id,receiver_user_id,sender_email,receiver_email,deal_id,company_id,is_draft,is_automated,message_id FROM emails",
"email_participants": "SELECT * FROM email_participants",
"email_reads": "SELECT message_id, read_at FROM email_reads",
"cloudtalk_sms": "SELECT id,text,direction,status,sender_user_id,created_date,company_id,phone_digits,is_echo,is_automated,media_urls IS NOT NULL AND media_urls<>'' AS has_media FROM cloudtalk_sms",
"ringcentral_sms": "SELECT id,text,direction,status,sender_user_id,created_date,company_id,phone_digits,is_echo,is_automated FROM ringcentral_sms",
"telnyx_sms": "SELECT id,text,direction,status,sender_user_id,created_date,company_id,phone_digits,is_echo,is_automated,delivery_status FROM telnyx_sms",
"provider_calls": """SELECT id,company_id,provider,customer_digits,started_at,
  JSON_UNQUOTE(JSON_EXTRACT(item,'$.Cdr.type')) type, JSON_UNQUOTE(JSON_EXTRACT(item,'$.Cdr.billsec')) billsec,
  JSON_UNQUOTE(JSON_EXTRACT(item,'$.Cdr.talking_time')) talking_time, JSON_UNQUOTE(JSON_EXTRACT(item,'$.Cdr.waiting_time')) waiting_time,
  JSON_UNQUOTE(JSON_EXTRACT(item,'$.Cdr.answered_at')) answered_at, JSON_UNQUOTE(JSON_EXTRACT(item,'$.Cdr.recorded')) recorded,
  JSON_UNQUOTE(JSON_EXTRACT(item,'$.Cdr.is_voicemail')) is_voicemail, JSON_UNQUOTE(JSON_EXTRACT(item,'$.Cdr.user_id')) agent_id,
  JSON_UNQUOTE(JSON_EXTRACT(item,'$.Agent.fullname')) agent_name, LEFT(item,400) item_head FROM provider_calls""",
"phone_calls": "SELECT id,company_id,user_id,direction,status,other_last10,customer_id,deal_id,started_at,answered_at,ended_at,duration_secs,hangup_cause,recording_secs,transcript,ai_note,spam_likely FROM phone_calls",
"telnyx_calls": "SELECT id,from_number,to_number,call_status,call_duration,transcript,created_at,reason FROM telnyx_calls",
"surveys": "SELECT * FROM customer_surveys",
"questionnaire_resp": "SELECT * FROM sales_questionnaire_responses",
"company": "SELECT id,name,address FROM company",
}
only = sys.argv[1:]
for k,q in Q.items():
    if only and k not in only: continue
    df = pd.read_sql(q, c)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(lambda v: v.decode('utf8','replace') if isinstance(v,(bytes,bytearray)) else v)
    df.to_parquet(f"data/raw/{k}.parquet")
    print(k, df.shape, flush=True)
